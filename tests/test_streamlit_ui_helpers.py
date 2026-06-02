from __future__ import annotations

import numpy as np
import pytest

from taf.ui.helpers import (
    MAX_MESSAGE_BITS,
    MIN_MESSAGE_BITS,
    apply_attacks,
    bits_to_string,
    calculate_bit_accuracy,
    generate_message,
    quick_attack_options,
    quick_metric_options,
    quick_method_options,
    string_to_bits,
    validate_binary_message,
)


def test_bits_to_string_and_string_to_bits_roundtrip() -> None:
    bits = [0, 1, 0, 1, 1, 0]

    value = bits_to_string(bits)

    assert value == "010110"
    assert string_to_bits(value) == bits


def test_validate_binary_message_accepts_boundaries() -> None:
    assert validate_binary_message("0" * MIN_MESSAGE_BITS) == [0] * MIN_MESSAGE_BITS
    assert validate_binary_message("1" * MAX_MESSAGE_BITS) == [1] * MAX_MESSAGE_BITS


@pytest.mark.parametrize("value", ["", "101", "0" * 121, "10a1", "10 1"])
def test_validate_binary_message_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_binary_message(value)


def test_generate_message_rejects_lengths_outside_boundaries() -> None:
    with pytest.raises(ValueError):
        generate_message(MIN_MESSAGE_BITS - 1)

    with pytest.raises(ValueError):
        generate_message(MAX_MESSAGE_BITS + 1)


def test_calculate_bit_accuracy_counts_missing_decoded_bits_as_incorrect() -> None:
    assert calculate_bit_accuracy([1, 0, 1, 1], [1, 1, 1]) == 0.5


def test_quick_method_options_include_all_registered_methods() -> None:
    from taf.methods.factory import SteganographyMethodFactory

    sample_rate = 16000

    option_types = {option.method_type for option in quick_method_options(sample_rate)}
    registered_types = set(SteganographyMethodFactory._all_methods(sample_rate))

    assert option_types == registered_types


def test_quick_metric_options_include_all_registered_metrics() -> None:
    from taf.metrics.factory import MetricFactory

    option_types = {option.metric_type for option in quick_metric_options()}
    registered_types = set(MetricFactory._all_methods())

    assert option_types == registered_types


def test_quick_attack_options_include_discovered_attacks() -> None:
    from taf.ui.helpers import discover_attacks

    option_names = {option.name for option in quick_attack_options()}
    discovered_names = {row["name"] for row in discover_attacks()}

    assert option_names == discovered_names


def test_apply_attacks_applies_selected_attacks() -> None:
    attack_options = [
        option for option in quick_attack_options() if option.name == "amplitude_scaling"
    ]
    samples = [0.1, -0.2, 0.3]

    attacked_samples, attacked_sample_rate = apply_attacks(
        samples=np.asarray(samples, dtype=np.float32),
        sample_rate=16000,
        attack_options=attack_options,
    )

    assert attacked_sample_rate == 16000
    assert attacked_samples.tolist() == pytest.approx([0.11, -0.22, 0.33])
