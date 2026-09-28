"""The sync-code scheme must find its blocks wherever they are.

These exercise the decoder's search, not robustness to signal processing: the
signal is only moved or cut, never altered, so a failure here means the sync
search itself is broken.
"""
from __future__ import annotations

import numpy as np
import pytest

from taf.methods.SyncDwtDctMethod import SyncDwtDctMethod
from taf.models.errors import CapacityError

SAMPLE_RATE = 16000


@pytest.fixture(scope="module")
def message() -> list[int]:
    return [int(bit) for bit in np.random.default_rng(606).integers(0, 2, 24)]


@pytest.fixture(scope="module")
def stego(speech_cover: np.ndarray, message: list[int]) -> np.ndarray:
    return SyncDwtDctMethod(SAMPLE_RATE).encode(speech_cover.copy(), message)


def _decode(samples: np.ndarray, length: int) -> list[int]:
    return [int(bit) for bit in SyncDwtDctMethod(SAMPLE_RATE).decode(samples, length)]


@pytest.mark.parametrize("delay", [1, 37, 1000, 5000])
def test_decodes_after_leading_silence(stego: np.ndarray, message: list[int], delay: int) -> None:
    delayed = np.concatenate((np.zeros(delay, dtype=stego.dtype), stego))
    assert _decode(delayed, len(message)) == message


@pytest.mark.parametrize("cut", [1, 999, 6000])
def test_decodes_after_the_start_is_cut_away(stego: np.ndarray, message: list[int], cut: int) -> None:
    # The message fits one block, so every block carries all of it.
    assert _decode(stego[cut:], len(message)) == message


def test_multi_block_message_survives_leading_silence(speech_cover: np.ndarray) -> None:
    method = SyncDwtDctMethod(SAMPLE_RATE, bits_per_block=16)
    message = [int(bit) for bit in np.random.default_rng(7).integers(0, 2, 40)]
    stego = method.encode(speech_cover.copy(), message)

    delayed = np.concatenate((np.zeros(2500, dtype=stego.dtype), stego))
    decoded = SyncDwtDctMethod(SAMPLE_RATE, bits_per_block=16).decode(delayed, len(message))
    assert [int(bit) for bit in decoded] == message


def test_no_block_is_found_in_an_unmarked_signal(speech_cover: np.ndarray) -> None:
    method = SyncDwtDctMethod(SAMPLE_RATE)
    assert method._find_blocks(np.asarray(speech_cover, dtype=np.float64), 24) == []


def test_capacity_is_blocks_times_bits_per_block(speech_cover: np.ndarray) -> None:
    method = SyncDwtDctMethod(SAMPLE_RATE)
    capacity = (len(speech_cover) // method.block_length) * method.bits_per_block
    method.encode(speech_cover.copy(), [1] * capacity)
    with pytest.raises(CapacityError):
        method.encode(speech_cover.copy(), [1] * (capacity + 1))
