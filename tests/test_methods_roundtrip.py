from __future__ import annotations

import numpy as np
import pytest

from taf.methods.factory import SteganographyMethodFactory
from taf.models.types import MethodType


@pytest.mark.parametrize("method_type", list(MethodType), ids=lambda m: m.name)
def test_method_encode_decode_does_not_crash(
    method_type: MethodType,
    sample_rate: int,
    synthetic_sine: np.ndarray,
    random_message: list[int],
) -> None:
    method = SteganographyMethodFactory.get(sample_rate, method_type)
    assert method is not None, f"factory returned None for {method_type}"

    try:
        encoded = method.encode(synthetic_sine.copy(), list(random_message))
    except ImportError as exc:
        pytest.skip(f"{method_type.name} requires an optional dependency: {exc}")

    assert isinstance(encoded, np.ndarray)
    assert encoded.size > 0

    decoded = method.decode(encoded, len(random_message))
    decoded_list = list(decoded)
    assert len(decoded_list) == len(random_message)
    assert all(int(bit) in {0, 1} for bit in decoded_list), (
        f"{method_type.name} decoded non-binary values: {decoded_list}"
    )


@pytest.mark.parametrize("method_type", list(MethodType), ids=lambda m: m.name)
def test_method_roundtrip_exact_on_speech(
    method_type: MethodType,
    sample_rate: int,
    speech_cover: np.ndarray,
    random_message: list[int],
) -> None:
    """Every method must return the message it was given, bit for bit.

    This used to be asserted for two methods only, with the rest merely
    required not to crash. That let real defects sit unnoticed: three methods
    were returning 18-56% of their bits wrong on ordinary speech, and one
    could only decode from the very object that had encoded.

    A fresh instance does the decoding, so nothing can be smuggled between the
    two halves on the instance itself.
    """
    method = SteganographyMethodFactory.get(sample_rate, method_type)

    try:
        encoded = method.encode(speech_cover.copy(), list(random_message))
    except ImportError as exc:
        pytest.skip(f"{method_type.name} requires an optional dependency: {exc}")

    decoder = SteganographyMethodFactory.get(sample_rate, method_type)
    decoded = [int(bit) for bit in decoder.decode(np.asarray(encoded), len(random_message))]

    assert decoded == random_message, (
        f"{method_type.name} failed exact roundtrip: "
        f"expected={random_message} decoded={decoded}"
    )


@pytest.mark.parametrize("method_type", list(MethodType), ids=lambda m: m.name)
def test_method_preserves_the_cover(
    method_type: MethodType,
    sample_rate: int,
    speech_cover: np.ndarray,
    random_message: list[int],
) -> None:
    """encode() must not write into the caller's array or change its length.

    The evaluation workflow keeps the cover to measure the stego against it.
    A method that embeds in place silently turns every quality metric into a
    comparison of the stego signal with itself.
    """
    method = SteganographyMethodFactory.get(sample_rate, method_type)
    cover = speech_cover.copy()
    reference = cover.copy()

    try:
        encoded = method.encode(cover, list(random_message))
    except ImportError as exc:
        pytest.skip(f"{method_type.name} requires an optional dependency: {exc}")

    np.testing.assert_array_equal(
        cover, reference, err_msg=f"{method_type.name} modified the cover in place"
    )
    assert len(encoded) == len(reference), (
        f"{method_type.name} changed the signal length: "
        f"{len(reference)} -> {len(encoded)}"
    )


@pytest.mark.parametrize("method_type", list(MethodType), ids=lambda m: m.name)
def test_method_rejects_an_over_capacity_message(
    method_type: MethodType,
    sample_rate: int,
    speech_cover: np.ndarray,
) -> None:
    """A message that does not fit must raise, not come back mangled.

    Several methods used to embed what fitted and drop the rest without a
    word, which reads as a very high bit error rate rather than as the
    capacity error it is.
    """
    method = SteganographyMethodFactory.get(sample_rate, method_type)
    # Well past the capacity of every frame-based method, while staying small
    # enough that the sample-domain ones, which really can carry it, finish
    # quickly.
    message = [1] * 5000

    try:
        encoded = method.encode(speech_cover.copy(), message)
    except ImportError as exc:
        pytest.skip(f"{method_type.name} requires an optional dependency: {exc}")
    except (ValueError, MemoryError):
        return

    decoded = [int(bit) for bit in method.decode(np.asarray(encoded), len(message))]
    assert decoded == message, (
        f"{method_type.name} silently truncated an over-capacity message "
        f"instead of raising ValueError"
    )
