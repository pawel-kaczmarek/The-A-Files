from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
src_path_text = str(SRC_PATH)

if src_path_text in sys.path:
    sys.path.remove(src_path_text)
sys.path.insert(0, src_path_text)


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


@pytest.fixture(scope="session")
def speech_cover() -> np.ndarray:
    """A real speech cover from the packaged VCTK subset.

    Synthetic tones are not a fair test bed for every method: a pure sine has
    a degenerate amplitude distribution, which the histogram method cannot
    embed into, so bit-exactness has to be asserted on real speech.
    """
    import soundfile as sf

    from taf.resources.paths import packaged_dataset_audio_paths

    with packaged_dataset_audio_paths() as dataset_paths:
        # The shortest clip keeps the slow neural methods tolerable.
        path = sorted(dataset_paths["vctk"])[2]
        samples, samplerate = sf.read(str(path), dtype="float32")

    assert samplerate == SAMPLE_RATE
    return samples
