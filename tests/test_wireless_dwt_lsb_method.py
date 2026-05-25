from __future__ import annotations

import numpy as np
import pytest

from taf.methods.WirelessDwtLsbMethod import WirelessDwtLsbMethod


def test_wireless_dwt_lsb_roundtrip_recovers_message() -> None:
    rng = np.random.default_rng(seed=20260523)
    t = np.arange(16000) / 16000.0
    cover = (0.4 * np.sin(2 * np.pi * 440.0 * t) + rng.normal(0.0, 0.005, len(t))).astype(np.float32)
    message = [int(bit) for bit in rng.integers(0, 2, size=257)]

    method = WirelessDwtLsbMethod()
    stego = method.encode(cover.copy(), message)
    decoded = method.decode(stego, len(message))

    assert stego.shape == cover.shape
    assert decoded == message


def test_wireless_dwt_lsb_roundtrip_recovers_message_from_int16_audio() -> None:
    rng = np.random.default_rng(seed=20260524)
    t = np.arange(16000) / 16000.0
    cover = np.rint(
        (0.4 * np.sin(2 * np.pi * 440.0 * t) + rng.normal(0.0, 0.005, len(t))) * 32767
    ).astype(np.int16)
    message = [int(bit) for bit in rng.integers(0, 2, size=257)]

    method = WirelessDwtLsbMethod()
    stego = method.encode(cover.copy(), message)
    decoded = method.decode(stego, len(message))

    assert stego.dtype == cover.dtype
    assert stego.shape == cover.shape
    assert decoded == message


def test_wireless_dwt_lsb_rejects_too_large_payload() -> None:
    cover = np.zeros(16, dtype=np.float32)
    method = WirelessDwtLsbMethod(level=1, lsb_depth=8)
    message = [1] * 128

    with pytest.raises(ValueError, match="message too long"):
        method.encode(cover, message)
