"""Reversible embedding must give back the cover exactly, not approximately."""
from __future__ import annotations

import numpy as np
import pytest

from taf.methods.ReversiblePeeMethod import ReversiblePeeMethod


def _message(length: int, seed: int = 11) -> list[int]:
    return [int(bit) for bit in np.random.default_rng(seed).integers(0, 2, length)]


@pytest.mark.parametrize("length", [8, 500, 3000])
def test_recovers_the_cover_sample_for_sample(speech_cover: np.ndarray, length: int) -> None:
    message = _message(length)
    stego = ReversiblePeeMethod().encode(speech_cover.copy(), message)

    assert not np.array_equal(stego, speech_cover)
    decoder = ReversiblePeeMethod()
    assert decoder.decode(stego, length) == message
    np.testing.assert_array_equal(decoder.recover_cover(stego, length), speech_cover)


def test_samples_near_full_scale_go_through_the_location_map(speech_cover: np.ndarray) -> None:
    """A loud, clipped cover would overflow without the location map."""
    # Rounded back onto the 16-bit grid, where recovery is exact.
    pcm = np.clip(np.round(speech_cover.astype(np.float64) * 4.5 * 32768), -32768, 32767)
    loud = (pcm / 32768).astype(np.float32)
    method = ReversiblePeeMethod(threshold=8)
    assert np.sum((pcm > 32767 - 8) | (pcm < -32768 + 8)) > 10, "the test cover must reach full scale"

    message = _message(2000)
    stego = method.encode(loud.copy(), message)

    assert np.all(np.abs(stego) <= 1.0)
    decoder = ReversiblePeeMethod(threshold=8)
    assert decoder.decode(stego, len(message)) == message
    np.testing.assert_array_equal(decoder.recover_cover(stego, len(message)), loud)


def test_only_the_prefix_the_payload_needs_is_changed(speech_cover: np.ndarray) -> None:
    stego = ReversiblePeeMethod().encode(speech_cover.copy(), _message(64))
    changed = np.nonzero(stego != speech_cover)[0]
    assert changed.max() < len(speech_cover) // 2


def test_a_damaged_signal_yields_bits_rather_than_an_exception(speech_cover: np.ndarray) -> None:
    message = _message(64)
    stego = ReversiblePeeMethod().encode(speech_cover.copy(), message)
    noisy = stego + np.random.default_rng(3).normal(0.0, 1e-3, len(stego)).astype(np.float32)

    decoded = ReversiblePeeMethod().decode(noisy, len(message))
    assert len(decoded) == len(message)
    assert set(decoded) <= {0, 1}
