from __future__ import annotations

import numpy as np
import pytest

from taf.methods.ImprovedSpreadSpectrumMethod import ImprovedSpreadSpectrumMethod
from taf.methods.LsbMethod import LsbMethod
from taf.steganalysis import (
    EnsembleClassifier,
    extract_features,
    feature_count,
    measure_detectability,
)

SAMPLE_RATE = 16000


def _speech_like(seed: int, seconds: float = 4.0) -> np.ndarray:
    """A voiced-sounding signal: harmonics with a wandering envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE

    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        signal += np.sin(2 * np.pi * 110.0 * harmonic * t) / harmonic

    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t + rng.uniform(0, np.pi))
    return (0.2 * envelope * signal + rng.normal(0, 0.005, t.shape)).astype(np.float32)


def test_extract_features_has_the_documented_length() -> None:
    features = extract_features(_speech_like(1))
    assert features.shape == (feature_count(),)
    assert np.all(np.isfinite(features))


def test_extract_features_is_deterministic() -> None:
    audio = _speech_like(2)
    np.testing.assert_allclose(extract_features(audio), extract_features(audio.copy()))


def test_extract_features_survives_non_finite_samples() -> None:
    """Silent or broken windows must not produce garbage array indices."""
    audio = _speech_like(3)
    audio[100] = np.nan
    audio[200] = np.inf

    features = extract_features(audio)
    assert np.all(np.isfinite(features))


def test_extract_features_rejects_a_too_short_signal() -> None:
    with pytest.raises(ValueError):
        extract_features(np.zeros(4, dtype=np.float32))


def test_ensemble_separates_two_distinct_clusters() -> None:
    rng = np.random.default_rng(20260921)
    covers = rng.normal(0.0, 1.0, size=(60, 40))
    stego = rng.normal(0.0, 1.0, size=(60, 40))
    stego[:, :5] += 4.0

    features = np.vstack([covers, stego])
    labels = np.array([0] * len(covers) + [1] * len(stego))

    model = EnsembleClassifier(base_learners=20, subspace_sizes=(5, 10)).fit(features, labels)
    assert model.score(features, labels) > 0.9
    assert model.oob_error < 0.2


def test_ensemble_rejects_a_single_class() -> None:
    features = np.random.default_rng(4).normal(size=(10, 6))
    with pytest.raises(ValueError):
        EnsembleClassifier(base_learners=5, subspace_sizes=(3,)).fit(
            features, np.zeros(10, dtype=np.int64)
        )


def test_ensemble_requires_fitting_before_use() -> None:
    with pytest.raises(ValueError):
        EnsembleClassifier().predict(np.zeros((2, 8)))


def test_measure_detectability_reports_a_usable_result() -> None:
    covers = [_speech_like(seed, seconds=6.0) for seed in range(4)]
    result = measure_detectability(
        ImprovedSpreadSpectrumMethod(strength=0.2),
        covers,
        message_length=8,
        window_length=SAMPLE_RATE * 2,
        classifier=EnsembleClassifier(base_learners=15, subspace_sizes=(5, 10)),
    )

    assert 0.0 <= result.accuracy <= 1.0
    assert result.test_size > 0
    assert result.train_size > result.test_size
    assert result.method


def test_measure_detectability_needs_enough_cover_material() -> None:
    with pytest.raises(ValueError):
        measure_detectability(LsbMethod(), [_speech_like(5, seconds=1.0)], window_length=SAMPLE_RATE)
