"""Tests for AAC + STC adaptive audio steganography (Luo et al. 2017)."""
from __future__ import annotations

import numpy as np
import pytest

from taf.methods.AacStcMethod import (
    AacStcMethod,
    _assign_costs,
    _make_h_hat,
    _stc_decode,
    _stc_encode,
)


@pytest.fixture
def audio_with_silence() -> np.ndarray:
    """1.024 s of mixed sine-active and silent regions @ 16 kHz."""
    sr = 16000
    n = 16384
    t = np.arange(n) / sr
    sine = 0.4 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    # Zero out the first quarter (mute region) and a chunk in the middle.
    sine[: n // 4] = 0.0
    sine[n // 2 : n // 2 + n // 8] = 0.0
    return sine


# -- Pure STC roundtrip ------------------------------------------------------


def test_stc_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    h, w = 7, 4
    k = 32
    n = k * w
    cover = rng.integers(0, 2, size=n).astype(np.uint8)
    message = rng.integers(0, 2, size=k).astype(np.uint8)
    cost = rng.uniform(0.1, 5.0, size=n)
    H = _make_h_hat(h, w, seed=42)

    e = _stc_encode(cover, message, cost, H)
    stego = cover ^ e
    recovered = _stc_decode(stego, H, k)

    assert recovered == list(message)


def test_stc_prefers_cheap_samples():
    """When some samples are expensive and the message is shifted by one block,
    STC should embed in the cheap region rather than the expensive one."""
    h, w = 6, 2
    k = 8
    n = k * w
    cover = np.zeros(n, dtype=np.uint8)
    message = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    cost = np.ones(n)
    cost[:n // 2] = 100.0           # first half: very expensive
    H = _make_h_hat(h, w, seed=7)

    e = _stc_encode(cover, message, cost, H)
    cheap_flips = int(e[n // 2:].sum())
    expensive_flips = int(e[: n // 2].sum())
    assert cheap_flips >= expensive_flips


# -- Cost assignment matches the paper --------------------------------------


def test_cost_assignment_matches_paper_equations():
    residual = np.array([-3, -1, 0, 1, 5], dtype=np.int32)
    cost_plus, cost_minus = _assign_costs(residual)
    # r<0 -> +1 is cheap (1/|r|), -1 is expensive (10/|r|)
    assert cost_plus[0] == pytest.approx(1 / 3)
    assert cost_minus[0] == pytest.approx(10 / 3)
    assert cost_plus[1] == pytest.approx(1.0)
    assert cost_minus[1] == pytest.approx(10.0)
    # r=0 -> 10 both ways
    assert cost_plus[2] == 10.0
    assert cost_minus[2] == 10.0
    # r>0 -> -1 is cheap, +1 is expensive
    assert cost_minus[3] == pytest.approx(1.0)
    assert cost_plus[3] == pytest.approx(10.0)
    assert cost_minus[4] == pytest.approx(1 / 5)
    assert cost_plus[4] == pytest.approx(10 / 5)


# -- Full method roundtrip ---------------------------------------------------


def test_method_roundtrip_recovers_message(audio_with_silence: np.ndarray):
    rng = np.random.default_rng(1)
    message = [int(b) for b in rng.integers(0, 2, size=32)]
    method = AacStcMethod(sr=16000)
    stego = method.encode(audio_with_silence.copy(), message)
    assert stego.shape == audio_with_silence.shape
    recovered = method.decode(stego, len(message))
    assert recovered == message


def test_perturbation_is_at_most_one_lsb(audio_with_silence: np.ndarray):
    """Method only modifies samples by +-1 in int16 space (~3e-5 in float)."""
    rng = np.random.default_rng(2)
    message = [int(b) for b in rng.integers(0, 2, size=16)]
    method = AacStcMethod(sr=16000)
    stego = method.encode(audio_with_silence.copy(), message)
    max_int_delta = int(
        np.max(np.abs(np.round(stego * 32767) - np.round(audio_with_silence * 32767)))
    )
    assert max_int_delta <= 1
