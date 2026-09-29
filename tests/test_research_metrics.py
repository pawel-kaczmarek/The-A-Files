import numpy as np
import pytest
from pystoi import stoi

from taf.metrics.speech_intelligibility.EstoiMetric import EstoiMetric
from taf.metrics.speech_quality.LogSpectralDistanceMetric import LogSpectralDistanceMetric
from taf.metrics.speech_quality.SpectralConvergenceMetric import SpectralConvergenceMetric
from taf.experiments.registry import list_metrics
from taf.plugins import create_metric


@pytest.mark.parametrize("metric", [LogSpectralDistanceMetric(), SpectralConvergenceMetric()])
def test_spectral_identity_and_gain(metric):
    x = np.random.default_rng(8).normal(0, .1, 16000)
    assert metric.calculate(x, x, 16000) == 0
    assert metric.calculate(x, .5*x, 16000) > 0
    if isinstance(metric, SpectralConvergenceMetric):
        assert metric.calculate(x, .5*x, 16000) == pytest.approx(.5)


def test_lsd_known_flat_spectrum_gain():
    x = np.zeros(512)
    x[256] = .9
    assert LogSpectralDistanceMetric().calculate(x, .5*x, 16000) == pytest.approx(20*np.log10(2))
    assert LogSpectralDistanceMetric().calculate(x*0, x*0, 16000) == 0


@pytest.mark.parametrize("metric", [LogSpectralDistanceMetric(), SpectralConvergenceMetric(), EstoiMetric()])
@pytest.mark.parametrize("x,y", [(np.zeros(10), np.zeros(11)), (np.zeros((20, 2)), np.zeros((20, 2))),
                                 (np.array([1, np.nan]), np.ones(2)), (np.zeros(0), np.zeros(0))])
def test_metrics_reject_invalid_signals(metric, x, y):
    with pytest.raises(ValueError):
        metric.calculate(x, y, 16000)


def test_silent_and_short_references_are_not_fake_scores():
    with pytest.raises(ValueError, match="silent"):
        SpectralConvergenceMetric().calculate(np.zeros(500), np.zeros(500), 16000)
    with pytest.raises(ValueError, match="non-silent"):
        EstoiMetric().calculate(np.zeros(16000), np.zeros(16000), 16000)
    with pytest.raises(RuntimeWarning):
        EstoiMetric().calculate(np.ones(500), np.ones(500), 16000)


def test_estoi_matches_extended_reference(speech_cover):
    noisy = speech_cover + np.random.default_rng(5).normal(0, .01, len(speech_cover))
    assert EstoiMetric().calculate(speech_cover, noisy, 16000) == pytest.approx(stoi(speech_cover, noisy, 16000, extended=True))


def test_new_metrics_registered_with_interpretation():
    entries = {row["name"]: row for row in list_metrics()}
    for name in ("ESTOI_METRIC", "LSD_METRIC", "MRSC_METRIC"):
        assert create_metric(name).name() == entries[name]["label"]
        assert entries[name]["details"]["en"] and entries[name]["details"]["pl"]


def test_visqol_optional_adapter_contract(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from taf.metrics.speech_quality.VisqolMetric import VisqolMetric

    model = tmp_path / "model" / "libsvm_nu_svr_model.txt"
    model.parent.mkdir()
    model.write_text("test model placeholder")
    calls = {}

    class Api:
        def Create(self, config):
            calls["config"] = config

        def Measure(self, x, y):
            calls["inputs"] = (x, y)
            return SimpleNamespace(moslqo=3.25)

    lib = SimpleNamespace(__file__=str(tmp_path / "visqol_lib_py.pyd"), VisqolApi=Api)
    pb = SimpleNamespace(VisqolConfig=lambda: SimpleNamespace(audio=SimpleNamespace(), options=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "visqol", SimpleNamespace(visqol_lib_py=lib))
    monkeypatch.setitem(sys.modules, "visqol.pb2", SimpleNamespace(visqol_config_pb2=pb))
    x = np.random.default_rng(5).normal(0, .1, 16000)
    assert VisqolMetric().calculate(x, x, 16000) == 3.25
    assert calls["config"].audio.sample_rate == 48000
    assert calls["config"].options.use_speech_scoring is False
    assert calls["config"].options.svr_model_path == str(model)
    assert calls["inputs"][0].shape == (48000,)
    np.testing.assert_array_equal(*calls["inputs"])
    model.unlink()
    with pytest.raises(ImportError, match="model is missing"):
        VisqolMetric().calculate(x, x, 16000)
