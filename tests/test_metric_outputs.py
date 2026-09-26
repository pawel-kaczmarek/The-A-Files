"""Every packaged metric runs at 16 kHz and declares what its numbers mean."""

from __future__ import annotations

import numpy as np
import pytest

from taf.metrics.factory import BUILTIN_METRICS
from taf.models.types import MetricType


@pytest.fixture(scope="module")
def signal_pair(speech_cover) -> tuple[np.ndarray, np.ndarray]:
    cover = np.asarray(speech_cover, dtype=np.float64)[: 16000 * 3]
    noisy = cover + np.random.default_rng(0).standard_normal(len(cover)) * 0.003
    return cover, noisy


@pytest.mark.parametrize("metric_type", list(MetricType), ids=lambda m: m.name)
def test_metric_declares_its_output(metric_type, signal_pair, sample_rate):
    metric = BUILTIN_METRICS[metric_type]()
    cover, processed = signal_pair
    try:
        value = metric.calculate(cover, processed, sample_rate, 0.03, 0.75)
    except ImportError as exc:
        pytest.skip(f"{metric_type.name} requires an optional dependency: {exc}")

    labelled = metric.labelled_values(value)
    directions = metric.directions()
    # Every value the engine records is named and has a declared direction
    # (None only for entries that are reported but never ranked).
    assert set(labelled) == set(directions), (
        f"{metric_type.name}: {np.asarray(value).size} value(s) but components {metric.components}"
    )
    assert any(direction is not None for direction in directions.values())
    assert any(np.isfinite(entry) for entry in labelled.values())


@pytest.mark.parametrize(
    "metric_type",
    [MetricType.SNR_METRIC, MetricType.STOI_METRIC, MetricType.STGI_METRIC, MetricType.WSTMI_METRIC],
    ids=lambda m: m.name,
)
def test_metric_orders_mild_before_strong_degradation(metric_type, speech_cover, sample_rate):
    cover = np.asarray(speech_cover, dtype=np.float64)[: 16000 * 3]
    noise = np.random.default_rng(1).standard_normal(len(cover))
    metric = BUILTIN_METRICS[metric_type]()
    mild = float(np.ravel(metric.calculate(cover, cover + 0.002 * noise, sample_rate))[0])
    strong = float(np.ravel(metric.calculate(cover, cover + 0.05 * noise, sample_rate))[0])
    assert (mild > strong) == metric.higher_is_better
