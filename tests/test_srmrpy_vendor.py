from __future__ import annotations

from taf.metrics.speech_reverberation.SrmrMetric import srmr


def test_vendored_srmrpy_imports() -> None:
    assert callable(srmr)
