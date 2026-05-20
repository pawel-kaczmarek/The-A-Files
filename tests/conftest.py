from __future__ import annotations

import numpy as np
import pytest


SAMPLE_RATE = 16000
# Echo/DSSS reserve floor(len/8192) frames and round down to a multiple of 8,
# so the signal must be long enough to yield at least 8 frames after rounding.
DURATION_SECONDS = 8
MESSAGE_LENGTH = 8


@pytest.fixture
def sample_rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def synthetic_sine() -> np.ndarray:
    rng = np.random.default_rng(seed=20260520)
    t = np.arange(SAMPLE_RATE * DURATION_SECONDS) / SAMPLE_RATE
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    noise = rng.normal(0.0, 0.01, size=t.shape)
    return (sine + noise).astype(np.float32)


@pytest.fixture
def random_message() -> list[int]:
    rng = np.random.default_rng(seed=20260520)
    return [int(b) for b in rng.integers(0, 2, size=MESSAGE_LENGTH)]
