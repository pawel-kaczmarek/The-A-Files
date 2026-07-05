"""Tests for the optional FastAPI layer (requires the ``api`` extra)."""

from __future__ import annotations

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from taf.api.app import create_app


@pytest.fixture()
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _wait_for_completion(client, experiment_id, timeout=180):
    deadline = time.monotonic() + timeout
    summary = None
    while time.monotonic() < deadline:
        summary = client.get(f"/api/experiments/{experiment_id}").json()
        if summary["status"] in {"completed", "failed"}:
            break
        time.sleep(0.5)
    return summary


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_catalog_endpoints(client):
    methods = client.get("/api/catalog/methods").json()
    metrics = client.get("/api/catalog/metrics").json()
    attacks = client.get("/api/catalog/attacks").json()
    datasets = client.get("/api/catalog/datasets").json()

    assert {"name", "class_name", "description", "requires_tensorflow"} <= set(methods[0])
    assert len(methods) >= 15
    assert len(metrics) >= 20
    assert len(attacks) >= 9
    # Attacks expose their parameters with defaults for the settings page.
    noise = next(row for row in attacks if row["name"] == "additive_noise")
    assert noise["parameters"] == [{"name": "std", "default": 0.001}]
    packaged = {row["id"] for row in datasets if row["kind"] == "packaged"}
    assert {"example", "vctk", "librispeech", "all"} <= packaged


def test_experiment_validation(client):
    base = {
        "experiment_type": "dataset_benchmark",
        "name": "bad",
        "dataset_id": "example",
        "methods": ["LSB_METHOD"],
        "payload_lengths": [8],
    }
    assert client.post("/api/experiments/run", json={**base, "methods": ["NOT_A_METHOD"]}).status_code == 422
    assert client.post("/api/experiments/run", json={**base, "attacks": ["not_an_attack"]}).status_code == 422
    assert client.post("/api/experiments/run", json={**base, "dataset_id": "upload:missing"}).status_code == 422
    # Scenario rule enforced at run time: robustness needs attacks.
    assert (
        client.post(
            "/api/experiments/run", json={**base, "experiment_type": "attack_robustness"}
        ).status_code
        == 422
    )


def test_preview_endpoint(client):
    response = client.post(
        "/api/experiments/preview",
        json={
            "experiment_type": "attack_robustness",
            "name": "preview",
            "dataset_id": "example",
            "methods": ["LSB_METHOD", "NORM_SPACE_METHOD"],
            "payload_lengths": [8, 16],
            "repetitions": 2,
            "attacks": ["additive_noise"],
        },
    )
    assert response.status_code == 200
    plan = response.json()
    assert plan["file_count"] == 1
    assert plan["attack_variant_count"] == 2
    assert plan["encode_operations"] == 8
    assert plan["estimated_result_rows"] == 16


def test_experiment_lifecycle(client):
    response = client.post(
        "/api/experiments/run",
        json={
            "experiment_type": "attack_robustness",
            "name": "lsb-robustness",
            "dataset_id": "example",
            "methods": ["LSB_METHOD"],
            "metrics": [],
            "payload_lengths": [8],
            "random_seed": 42,
            "attacks": ["amplitude_scaling"],
        },
    )
    assert response.status_code == 201
    body = response.json()
    experiment_id = body["experiment_id"]
    assert body["attacks"] == ["amplitude_scaling"]
    assert body["csv_url"].endswith("/export.csv")

    summary = _wait_for_completion(client, experiment_id)
    assert summary is not None
    assert summary["status"] == "completed", summary.get("error")
    assert summary["completed_rows"] == 2  # baseline + attack
    assert summary["total_tasks"] == 2

    rows = client.get(f"/api/experiments/{experiment_id}/results").json()
    assert len(rows) == 2
    assert {row["attack"] for row in rows} == {None, "amplitude_scaling"}
    assert all(row["ber"] is not None for row in rows)

    report = client.get(f"/api/experiments/{experiment_id}/summary").json()
    assert report["summary"]["most_robust_method"]

    csv_response = client.get(f"/api/experiments/{experiment_id}/export.csv")
    assert csv_response.status_code == 200
    header = csv_response.text.splitlines()[0]
    for column in ("experiment_id", "method", "attack", "ber", "bit_accuracy", "status"):
        assert column in header
    assert "attack_robustness_" in csv_response.headers["content-disposition"]

    summary_csv = client.get(f"/api/experiments/{experiment_id}/export_summary.csv")
    assert summary_csv.status_code == 200

    config_json = client.get(f"/api/experiments/{experiment_id}/config.json").json()
    assert config_json["experiment_type"] == "attack_robustness"

    listing = client.get("/api/experiments/history").json()
    assert any(item["experiment_id"] == experiment_id for item in listing)


def test_experiment_not_found(client):
    assert client.get("/api/experiments/does-not-exist").status_code == 404
    assert client.get("/api/experiments/does-not-exist/summary").status_code == 404
    assert client.get("/api/experiments/does-not-exist/results").status_code == 404


def test_global_event_stream_snapshot(client):
    response = client.post(
        "/api/experiments/run",
        json={
            "experiment_type": "dataset_benchmark",
            "name": "stream-smoke",
            "dataset_id": "example",
            "methods": ["LSB_METHOD"],
            "payload_lengths": [8],
            "random_seed": 1,
        },
    )
    assert response.status_code == 201
    experiment_id = response.json()["experiment_id"]
    summary = _wait_for_completion(client, experiment_id)
    assert summary["status"] == "completed", summary.get("error")

    # max_events=0 ends the stream right after the snapshot event, so the
    # response can be read to completion (aborting an infinite SSE body hangs
    # the TestClient).
    response = client.get("/api/experiments/events", params={"max_events": 0})
    assert response.status_code == 200
    lines = response.text.splitlines()
    assert lines[0] == "event: experiments"
    snapshot = json.loads(lines[1].removeprefix("data: "))
    entry = next(item for item in snapshot if item["experiment_id"] == experiment_id)
    assert entry["status"] == "completed"


def test_uploaded_dataset_lifecycle(client, tmp_path):
    import shutil

    from taf.resources.paths import example_wav_path

    with example_wav_path() as source:
        local_copy = tmp_path / "own_sound.wav"
        shutil.copy(source, local_copy)

    with local_copy.open("rb") as handle:
        response = client.post(
            "/api/datasets/uploads",
            data={"name": "my sounds"},
            files=[("files", ("own_sound.wav", handle, "audio/wav"))],
        )
    assert response.status_code == 201
    upload = response.json()
    assert upload["file_count"] == 1
    assert upload["dataset_name"].startswith("upload:")

    datasets = client.get("/api/catalog/datasets").json()
    assert any(row["id"] == upload["dataset_name"] for row in datasets)

    response = client.post(
        "/api/experiments/run",
        json={
            "experiment_type": "dataset_benchmark",
            "name": "upload-smoke",
            "dataset_id": upload["dataset_name"],
            "methods": ["LSB_METHOD"],
            "payload_lengths": [8],
            "random_seed": 7,
        },
    )
    assert response.status_code == 201
    experiment_id = response.json()["experiment_id"]
    summary = _wait_for_completion(client, experiment_id)
    assert summary["status"] == "completed", summary.get("error")
    rows = client.get(f"/api/experiments/{experiment_id}/results").json()
    assert rows[0]["decode_success"] is True
    assert rows[0]["file_name"] == "own_sound.wav"

    assert client.delete(f"/api/datasets/uploads/{upload['id']}").status_code == 204
    assert client.delete(f"/api/datasets/uploads/{upload['id']}").status_code == 404


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/datasets/uploads",
        files=[("files", ("notes.txt", b"hello", "text/plain"))],
    )
    assert response.status_code == 422
