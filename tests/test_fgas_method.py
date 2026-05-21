"""Tests for the FGAS (arXiv:2505.22266) audio steganography method.

FGAS is a deep, gradient-based method. Its bit-recovery is not algebraically
exact; the paper itself reports >99% (not 100%) at moderate payloads, and
recovery quality depends on the cover signal, payload and iteration budget.
These tests assert FGAS's design contracts rather than perfect roundtrip:

* the stego waveform shape matches the cover
* the L_inf perturbation budget (epsilon) is honoured
* decoding is sensitive to the shared key
* recovery on a moderate payload beats random guessing comfortably
"""
from __future__ import annotations

import numpy as np
import pytest

from taf.methods.FgasMethod import FgasMethod


@pytest.fixture
def short_audio() -> np.ndarray:
    rng = np.random.default_rng(seed=20260521)
    t = np.arange(16000) / 16000.0  # 1 second @ 16 kHz
    sine = 0.4 * np.sin(2 * np.pi * 440.0 * t)
    noise = rng.normal(0.0, 0.005, size=t.shape)
    return (sine + noise).astype(np.float32)


def test_fgas_respects_perturbation_budget(short_audio: np.ndarray) -> None:
    rng = np.random.default_rng(seed=1)
    message = [int(b) for b in rng.integers(0, 2, size=16)]

    method = FgasMethod(sr=16000, iterations=80, epsilon=1e-3)
    stego = method.encode(short_audio.copy(), message)

    assert isinstance(stego, np.ndarray)
    assert stego.shape == short_audio.shape
    assert np.max(np.abs(stego - short_audio)) <= method.epsilon + 1e-6


def test_fgas_recovers_with_practical_epsilon(short_audio: np.ndarray) -> None:
    """With a sufficient L_inf budget, APG should recover all bits.

    The paper's ε=1e-3 default reaches ~99% only with 2000 iterations and the
    anti-detection loss term. With our compact test budget (200 iters,
    BCE-only loss) we relax ε to 5e-2 — still well below an audibly
    significant perturbation for normalized speech — and demand exact
    recovery.
    """
    rng = np.random.default_rng(seed=2)
    message = [int(b) for b in rng.integers(0, 2, size=32)]

    method = FgasMethod(sr=16000, hidden_channels=16, iterations=200, epsilon=5e-2)
    stego = method.encode(short_audio.copy(), message)
    decoded = method.decode(stego, len(message))

    assert len(decoded) == len(message)
    assert all(b in (0, 1) for b in decoded)
    assert decoded == message


def test_fgas_wrong_key_loses_information(short_audio: np.ndarray) -> None:
    rng = np.random.default_rng(seed=3)
    message = [int(b) for b in rng.integers(0, 2, size=32)]

    sender = FgasMethod(sr=16000, hidden_channels=16, key=42, iterations=200, epsilon=5e-2)
    stego = sender.encode(short_audio.copy(), message)

    correct = sender.decode(stego, len(message))
    correct_matches = sum(1 for a, b in zip(correct, message) if a == b)

    eavesdropper = FgasMethod(sr=16000, hidden_channels=16, key=9999, iterations=200, epsilon=5e-2)
    wrong = eavesdropper.decode(stego, len(message))
    wrong_matches = sum(1 for a, b in zip(wrong, message) if a == b)

    # The correct-key decoder must do meaningfully better than a decoder using
    # the wrong key (the security guarantee of the shared-key construction).
    assert correct_matches > wrong_matches
