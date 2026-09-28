from __future__ import annotations

import numpy as np

from taf.methods.common.emd import emd
from taf.methods.EmdMethod import EmdMethod


def test_decomposition_adds_up_to_the_signal(speech_cover: np.ndarray) -> None:
    frame = np.asarray(speech_cover[20000:21024], dtype=np.float64)
    imfs, residue = emd(frame, 4)

    assert len(imfs) == 4
    np.testing.assert_allclose(np.sum(imfs, axis=0) + residue, frame, atol=1e-12)


def test_decomposition_is_deterministic(speech_cover: np.ndarray) -> None:
    frame = np.asarray(speech_cover[20000:21024], dtype=np.float64)
    first, _ = emd(frame, 4)
    second, _ = emd(frame.copy(), 4)
    for one, other in zip(first, second):
        np.testing.assert_array_equal(one, other)


def test_payload_survives_a_power_of_two_volume_change(speech_cover: np.ndarray) -> None:
    """The step follows the signal level, so halving the volume changes nothing.

    A factor of two scales every float exactly, which isolates the step
    definition from rounding.
    """
    message = [int(bit) for bit in np.random.default_rng(21).integers(0, 2, 16)]
    stego = EmdMethod().encode(speech_cover.copy(), message)

    assert EmdMethod().decode(stego * np.float32(0.5), len(message)) == message
