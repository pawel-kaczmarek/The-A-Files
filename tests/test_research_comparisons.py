from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from taf.experiments.research import research_comparison
from taf.experiments.results import ExperimentResultRow
from taf.methods.literature import PAPERS


def row(**kwargs):
    values = dict(experiment_id="test", experiment_type="dataset_benchmark", timestamp=datetime.now(timezone.utc),
                  file_name="same.wav", file_path="a/same.wav", file_id="a/same.wav", method="LSB", payload_length=16,
                  payload_kind="random", audio_category="speech", ber=0, decode_success=True,
                  payload_rate_bps=8., exact_goodput_bps=8., encode_rtf=.01, metrics={"LSD": 0.1})
    return ExperimentResultRow(**{**values, **kwargs})


def test_separate_conditions_failures_and_file_clusters():
    rows = [row(), row(file_path="b/same.wav", file_id="b/same.wav", ber=None, decode_success=False,
                       status="error", exact_goodput_bps=0., audio_category="music"),
            row(attack="noise", ber=.25, exact_goodput_bps=0., decode_success=False, attack_metrics={"LSD": 9.})]
    baseline = research_comparison(rows)
    values = baseline["groups"][0]["measures"]
    assert baseline["rows"] == 2
    assert values["ber"]["count"] == 1
    assert values["exact_goodput_bps"]["mean"] == 4
    assert values["exact_goodput_bps"]["clusters"] == 2
    attacked = research_comparison(rows, attack="noise", reference="attack")
    assert attacked["groups"][0]["measures"]["quality:LSD"]["mean"] == 9
    filtered = research_comparison(rows, filters={"audio_category": "music"}, timing_reliable=False)
    assert filtered["rows"] == 1
    assert "encode_rtf" not in filtered["groups"][0]["measures"]


def test_invalid_comparison_dimensions():
    with pytest.raises(ValueError):
        research_comparison([], group="missing")
    with pytest.raises(ValueError):
        research_comparison([], reference="mixed")


def test_rate_capacity_does_not_claim_a_shared_bit_length_grid():
    from taf.experiments.schema import ExperimentConfig
    from taf.experiments.scenarios.embedding_capacity import _summarize

    config = ExperimentConfig(name="rate capacity", experiment_type="embedding_capacity", dataset_id="vctk",
                              methods=["LSB_METHOD"], payload_rates_bps=[8, 16])
    rows = [row(file_id=file, duration_seconds=duration, payload_length=int(rate * duration),
                payload_rate_bps=rate, bit_accuracy=1.)
            for file, duration in (("short.wav", 1.), ("long.wav", 2.)) for rate in (8., 16.)]
    result = _summarize(rows, config)
    capacity = result["capacity_by_method"][0]
    assert capacity["capacity_bps_mean"] == 16.
    assert capacity["censored_files"] == 2
    assert capacity["max_passing_payload"] is None
    assert not capacity["pooled_payload_grid_comparable"]
    assert result["highest_stable_payload"] is None


def test_literature_has_auditable_evidence():
    assert len({p.id for p in PAPERS}) == len(PAPERS)
    for p in PAPERS:
        assert p.authors and p.doi and p.url.startswith("https://")
        assert p.datasets and p.metrics and p.attacks and p.payload
        assert p.limitations and p.source_url.startswith("https://")
        for result in p.results:
            assert np.isfinite(result.value) and result.condition and result.locator
    assert all(not p.implemented_methods for p in PAPERS)


def test_catalog_and_comparison_http(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlalchemy")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from taf.api.routers import catalog, runs

    app = FastAPI()
    app.include_router(catalog.router)
    app.include_router(runs.router)
    monkeypatch.setattr(runs, "_require", lambda _: (SimpleNamespace(config={"max_workers": 1}), None))
    monkeypatch.setattr(runs.store, "all_rows", lambda _: [row(audio_source="Test corpus", requested_payload_rate_bps=8.)])
    client = TestClient(app)
    papers = client.get("/api/catalog/literature?purpose=steganography").json()
    assert [p["id"] for p in papers] == ["hide-and-speak"]
    assert client.get("/api/catalog/literature?q=FMA&year=2024").json()[0]["id"] == "ideaw"
    url = f"/api/runs/{uuid.uuid4()}/research"
    response = client.get(url)
    assert response.status_code == 200 and response.json()["rows"] == 1
    assert client.get(url + "?audio_category=music").json()["rows"] == 0
    assert client.get(url + "?group=missing").status_code == 422
    assert client.get(url, params={"requested_payload_rate_bps": "8", "audio_source": "Test corpus"}).json()["rows"] == 1
    assert client.get(url, params={"requested_payload_rate_bps": "16"}).json()["rows"] == 0


def test_attack_quality_retained_when_decoder_fails(tmp_path):
    from taf.evaluation import EvaluationConfig, evaluate_files
    from taf.models.SteganographyMethod import SteganographyMethod
    from taf.models.WavFile import WavFile

    class Broken(SteganographyMethod):
        def encode(self, data, message):
            return data.copy()

        def decode(self, data_with_watermark, watermark_length):
            raise RuntimeError("decode failed")

        def type(self):
            return "Broken"

    signal = np.random.default_rng(5).normal(0, .1, 16000)
    result = evaluate_files([WavFile(16000, signal, tmp_path / "audio.wav")], EvaluationConfig(
        methods=[lambda _: Broken()], metrics=["LSD_METRIC"], random_message_lengths=[16], random_seed=5,
        attacks=["gain:gain_db=-6"], keep_files=False,
    ))
    clean, attacked = result.rows
    assert attacked.failure_kind == "decode_error"
    assert attacked.metrics == clean.metrics
    assert list(attacked.attack_metrics.values())[0] > 0
