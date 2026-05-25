from __future__ import annotations

import numpy as np
import pytest

from taf.methods.factory import SteganographyMethodFactory
from taf.models.types import MethodType


# Methods we expect to recover bits exactly on synthetic audio at 16 kHz.
# Other methods are research code with known sensitivity to signal content;
# they are exercised by the no-crash test below but not asserted bit-exact here.
STRICT_METHODS = [MethodType.LSB_METHOD, MethodType.WIRELESS_DWT_LSB_METHOD]


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


@pytest.mark.parametrize("method_type", STRICT_METHODS, ids=lambda m: m.name)
def test_method_roundtrip_exact(
    method_type: MethodType,
    sample_rate: int,
    synthetic_sine: np.ndarray,
    random_message: list[int],
) -> None:
    method = SteganographyMethodFactory.get(sample_rate, method_type)
    encoded = method.encode(synthetic_sine.copy(), list(random_message))
    decoded = [int(bit) for bit in method.decode(encoded, len(random_message))]
    assert decoded == random_message, (
        f"{method_type.name} failed exact roundtrip: expected={random_message} decoded={decoded}"
    )
