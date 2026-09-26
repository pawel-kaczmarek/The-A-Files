"""The research platform API over PostgreSQL (requires the ``platform`` extra).

The tests use their own database, ``TAF_TEST_DATABASE_URL`` (default: a
``taf_test`` database on the server of ``docker-compose.yml``), whose schema
is dropped and migrated from scratch; they are skipped when no PostgreSQL
server is reachable.
"""

from __future__ import annotations

import io
import os
import tarfile
import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

import soundfile as sf
from fastapi.testclient import TestClient

TEST_DATABASE_URL = os.environ.get(
    "TAF_TEST_DATABASE_URL", "postgresql+psycopg://taf:taf@localhost:5432/taf_test"
)


def _prepare_database(url: str) -> None:
    import sqlalchemy as sa
    from sqlalchemy.engine import make_url

    target = make_url(url)
    admin = sa.create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        exists = connection.execute(
            sa.text("select 1 from pg_database where datname = :name"), {"name": target.database}
        ).scalar()
        if not exists:
            connection.execute(sa.text(f'create database "{target.database}"'))
    admin.dispose()
    engine = sa.create_engine(url, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(sa.text("drop schema public cascade"))
        connection.execute(sa.text("create schema public"))
    engine.dispose()


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    from sqlalchemy.exc import OperationalError

    try:
        _prepare_database(TEST_DATABASE_URL)
    except OperationalError as error:
        pytest.skip(f"PostgreSQL not available: {str(error).splitlines()[0]}")

    from taf.persistence import session

    previous = {key: os.environ.get(key) for key in ("TAF_DATABASE_URL", "TAF_DATA_DIR")}
    os.environ["TAF_DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["TAF_DATA_DIR"] = str(tmp_path_factory.mktemp("taf-data"))
    session.reset_caches()
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    session.reset_caches()


@pytest.fixture(scope="module")
def client(database):
    from taf.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def _wait(client, url: str, done, timeout: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(url).json()
        if done(body):
            return body
        time.sleep(0.3)
    raise AssertionError(f"timed out waiting on {url}: {body}")


def _experiment(client, **overrides) -> dict:
    payload = {
        "name": "LSB vs QIM",
        "experiment_type": "dataset_benchmark",
        "research_question": "Does QIM decode more reliably than LSB?",
        "hypothesis": "QIM has a lower BER than LSB.",
        "tags": ["test"],
        "config": {
            "dataset_id": "vctk",
            "file_limit": 2,
            "methods": ["LSB_METHOD", "QIM_METHOD"],
            "metrics": ["SNR_METRIC"],
            "payload_lengths": [8],
            "random_seed": 3,
        },
    }
    payload.update(overrides)
    response = client.post("/api/experiments", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _run(client, experiment_id: str) -> dict:
    response = client.post(f"/api/experiments/{experiment_id}/runs")
    assert response.status_code == 201, response.text
    run = response.json()
    return _wait(client, f"/api/runs/{run['id']}", lambda body: body["status"] not in ("queued", "running"))


# ------------------------------------------------------------------ system


def test_health_reports_the_database(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["database"]["ok"] and "PostgreSQL" in body["database"]["version"]


def test_catalog_describes_components(client):
    methods = {row["name"]: row for row in client.get("/api/catalog/methods").json()}
    assert methods["QIM_METHOD"]["family"] == "quantization"
    assert methods["QIM_METHOD"]["strength_parameter"] == "step_scale"
    assert any(p["name"] == "step_scale" for p in methods["QIM_METHOD"]["parameters"])

    metrics = {row["name"]: row for row in client.get("/api/catalog/metrics").json()}
    assert metrics["SNR_METRIC"]["higher_is_better"] is True
    assert metrics["WSS_METRIC"]["higher_is_better"] is False
    assert metrics["BSS_EVAL_METRIC"]["components"] == ["sdr", "isr", "sir", "sar", "perm"]

    attacks = {row["name"]: row for row in client.get("/api/catalog/attacks").json()}
    assert attacks["awgn"]["family"] == "noise" and attacks["awgn"]["stochastic"]
    assert attacks["mp3"]["sweep"]["parameter"] == "bitrate_kbps"

    designs = {row["type"]: row for row in client.get("/api/catalog/designs").json()}
    assert designs["robustness_curve"]["requires_sweep"] == "attack"
    assert {row["property"] for row in designs.values()} >= {"imperceptibility", "robustness", "capacity", "security"}

    corpora = {row["id"]: row for row in client.get("/api/catalog/corpora").json()}
    assert corpora["librispeech_test_clean"]["downloadable"] and not corpora["timit"]["downloadable"]

    presets = client.get("/api/catalog/presets", params={"sample_rate": 16000}).json()
    assert presets["sweeps"]["low_pass"]["values"][0] < 8000


# ------------------------------------------------------------- experiments


def test_experiment_versioning_and_lifecycle(client):
    experiment = _experiment(client)
    assert experiment["version"] == 1 and experiment["problems"] == []

    renamed = client.put(
        f"/api/experiments/{experiment['id']}",
        json={**{k: experiment[k] for k in ("name", "experiment_type", "research_question", "hypothesis", "tags", "config")}, "name": "Renamed"},
    ).json()
    assert renamed["name"] == "Renamed" and renamed["version"] == 1  # metadata only

    changed = client.put(
        f"/api/experiments/{experiment['id']}",
        json={**{k: renamed[k] for k in ("name", "experiment_type", "research_question", "hypothesis", "tags")}, "config": {**renamed["config"], "payload_lengths": [8, 16]}},
    ).json()
    assert changed["version"] == 2  # the protocol changed

    copy = client.post(f"/api/experiments/{experiment['id']}/duplicate").json()
    assert copy["id"] != experiment["id"] and copy["config"] == changed["config"]

    client.post(f"/api/experiments/{copy['id']}/archive")
    assert all(row["id"] != copy["id"] for row in client.get("/api/experiments").json())
    assert any(row["id"] == copy["id"] for row in client.get("/api/experiments", params={"include_archived": True}).json())
    assert client.delete(f"/api/experiments/{copy['id']}").status_code == 204
    assert client.get(f"/api/experiments/{copy['id']}").status_code == 404


def test_invalid_protocols_are_rejected_or_flagged(client):
    bad = client.post(
        "/api/experiments",
        json={"name": "bad", "experiment_type": "dataset_benchmark", "config": {"dataset_id": "vctk", "methods": ["NOPE"]}},
    )
    assert bad.status_code == 422

    incomplete = _experiment(client, name="curve without sweep", experiment_type="robustness_curve")
    assert any("attack sweep" in problem for problem in incomplete["problems"])
    assert client.post(f"/api/experiments/{incomplete['id']}/runs").status_code == 422


# -------------------------------------------------------------------- runs


def test_run_is_persisted_reported_and_inspectable(client):
    experiment = _experiment(client, config={
        "dataset_id": "vctk",
        "file_limit": 2,
        "methods": ["LSB_METHOD", "QIM_METHOD:step_scale=0.2"],
        "metrics": ["SNR_METRIC"],
        "attacks": ["awgn:snr_db=20"],
        "payload_lengths": [8],
    })
    run = _run(client, experiment["id"])
    assert run["status"] == "completed", run["error"]
    assert run["completed_rows"] == run["total_rows"] == 2 * 2 * 2

    page = client.get(f"/api/runs/{run['id']}/rows", params={"limit": 3}).json()
    assert page["total"] == 8 and len(page["rows"]) == 3
    baseline = client.get(f"/api/runs/{run['id']}/rows", params={"attack": ""}).json()
    assert baseline["total"] == 4 and all(r["row"]["attack"] is None for r in baseline["rows"])
    facets = client.get(f"/api/runs/{run['id']}/facets").json()
    assert "Quantization index modulation (spread-transform dither modulation) (step_scale=0.2)" in facets["method"]

    summary = client.get(f"/api/runs/{run['id']}/summary").json()["summary"]
    assert "statistics" in summary and "pareto" in summary

    config = client.get(f"/api/runs/{run['id']}/config.json").json()
    assert isinstance(config["random_seed"], int)  # drawn at start and recorded
    manifest = client.get(f"/api/runs/{run['id']}/manifest.json").json()
    assert manifest["random_seed"] == config["random_seed"]
    assert "file_name" in client.get(f"/api/runs/{run['id']}/export.csv").text
    report = client.get(f"/api/runs/{run['id']}/report.md").text
    assert "## Experimental setup" in report and "seed" in report
    assert "\\begin{tabular}" in client.get(f"/api/runs/{run['id']}/report.tex").text

    attacked = client.get(f"/api/runs/{run['id']}/rows", params={"attack": "awgn:snr_db=20"}).json()["rows"][0]
    inspection = client.get(f"/api/runs/{run['id']}/rows/{attacked['id']}/inspect").json()
    assert inspection["reproduced"] is True
    assert set(inspection["signals"]) == {"cover", "stego", "attacked", "residual"}
    audio = client.get(f"/api/runs/{run['id']}/rows/{attacked['id']}/audio/residual.wav")
    assert audio.headers["content-type"] == "audio/wav" and audio.content[:4] == b"RIFF"

    events = client.get(f"/api/runs/{run['id']}/events").text
    assert events.startswith("event: snapshot") and "event: done" in events

    runs = client.get(f"/api/experiments/{experiment['id']}/runs").json()
    assert [r["number"] for r in runs] == [1]
    assert client.post(f"/api/runs/{run['id']}/cancel").status_code == 409


def test_robustness_curve_through_the_api(client):
    experiment = _experiment(client, name="AWGN curve", experiment_type="robustness_curve", config={
        "dataset_id": "vctk",
        "file_limit": 2,
        "methods": ["QIM_METHOD", "LSB_METHOD"],
        "payload_lengths": [8],
        "attack_sweep": {"target": "awgn", "parameter": "snr_db", "values": [30, 10, 0]},
    })
    run = _run(client, experiment["id"])
    assert run["status"] == "completed", run["error"]
    summary = client.get(f"/api/runs/{run['id']}/summary").json()["summary"]
    assert [point["value"] for point in summary["curves"][0]["points"]] == [30, 10, 0]


# ---------------------------------------------------------------- datasets


def _wav_bytes(seconds: float = 2.0, seed: int = 0) -> bytes:
    buffer = io.BytesIO()
    signal = 0.1 * np.random.default_rng(seed).standard_normal(int(16000 * seconds))
    sf.write(buffer, signal, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_synthetic_dataset_is_usable_by_experiments(client):
    dataset = client.post("/api/datasets/synthetic", json={"duration_seconds": 3.0}).json()
    assert dataset["status"] == "ready" and dataset["file_count"] == 10
    ids = [row["id"] for row in client.get("/api/catalog/datasets").json()]
    assert f"library:{dataset['id']}" in ids

    experiment = _experiment(client, name="edge cases", config={
        "dataset_id": f"library:{dataset['id']}",
        "file_limit": 3,
        "methods": ["LSB_METHOD"],
        "payload_lengths": [8],
    })
    run = _run(client, experiment["id"])
    assert run["status"] == "completed", run["error"]
    assert run["completed_rows"] == 3


def test_upload_and_local_registration(client, tmp_path):
    upload = client.post(
        "/api/datasets/upload",
        data={"name": "my recordings"},
        files=[("files", ("one.wav", _wav_bytes(), "audio/wav")), ("files", ("two.wav", _wav_bytes(seed=1), "audio/wav"))],
    )
    assert upload.status_code == 201, upload.text
    assert upload.json()["file_count"] == 2
    assert client.post(
        "/api/datasets/upload", files=[("files", ("notes.txt", b"hi", "text/plain"))]
    ).status_code == 422

    nested = tmp_path / "corpus" / "speaker1"
    nested.mkdir(parents=True)
    (nested / "a.wav").write_bytes(_wav_bytes())
    local = client.post(
        "/api/datasets/local", json={"name": "TIMIT copy", "path": str(tmp_path / "corpus"), "corpus_id": "timit"}
    ).json()
    assert local["kind"] == "local" and local["file_count"] == 1 and "LDC" in local["license"]

    assert client.delete(f"/api/datasets/{upload.json()['id']}").status_code == 204
    assert client.get(f"/api/datasets/{upload.json()['id']}").status_code == 404


def test_corpus_preparation_from_a_local_archive(client, tmp_path):
    archive = tmp_path / "mini.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for speaker in ("11", "22", "33"):
            for utterance in range(3):
                data = _wav_bytes(seconds=2.0, seed=int(speaker) + utterance)
                member = tarfile.TarInfo(f"LibriSpeech/dev-clean-2/{speaker}/100/{speaker}-100-{utterance}.flac")
                buffer = io.BytesIO()
                samples, rate = sf.read(io.BytesIO(data))
                sf.write(buffer, samples, rate, format="FLAC")
                member.size = len(buffer.getvalue())
                tar.addfile(member, io.BytesIO(buffer.getvalue()))

    response = client.post(
        "/api/datasets/prepare",
        json={"corpus_id": "mini_librispeech", "source_path": str(archive), "rule": {"max_files": 4, "seed": 1}},
    )
    assert response.status_code == 202, response.text
    dataset = _wait(client, f"/api/datasets/{response.json()['id']}", lambda body: body["status"] in ("ready", "failed"))
    assert dataset["status"] == "ready", dataset["error"]
    assert dataset["file_count"] == 4
    assert dataset["manifest"]["speakers"] == 3  # balanced over speakers
    assert all(len(entry["sha256"]) == 64 for entry in dataset["manifest"]["files"])

    assert client.post("/api/datasets/prepare", json={"corpus_id": "timit"}).status_code == 422
