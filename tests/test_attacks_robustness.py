"""End-to-end robustness tests: embed, attack, extract, measure.

These check that the *experiment pipeline* is sound - that an attack can be
applied between encoding and decoding, that the bit error rate is computed
over the right bits, and that the attack's parameters reach the result row.

They deliberately do not require any method to survive any attack. A method
that fails an attack is a scientific result, not a test failure; asserting
otherwise would create pressure to weaken the benchmark until the methods look
good, which is exactly what a benchmark must not do. The only robustness
assertions here are the two that must hold for the measurement itself to mean
anything: a payload survives when nothing is done to it, and a destructive
attack does measurable damage.
"""
from __future__ import annotations

import numpy as np
import pytest

from taf.attacks import build
from taf.attacks.codec import ffmpeg_available
from taf.experiments.results import bit_error_rate
from taf.methods.factory import SteganographyMethodFactory
from taf.models.types import MethodType

SAMPLE_RATE = 16000
MESSAGE_LENGTH = 16

# A spread of embedding domains: sample-domain, transform-domain with a
# relation-based detector, quantisation-based, and spread spectrum.
METHODS = [
    MethodType.LSB_METHOD,
    MethodType.QIM_METHOD,
    MethodType.IMPROVED_SPREAD_SPECTRUM_METHOD,
    MethodType.DWT_LSB_METHOD,
]


@pytest.fixture(scope="module")
def message() -> list[int]:
    return [int(bit) for bit in np.random.default_rng(4242).integers(0, 2, MESSAGE_LENGTH)]


def _stego(method_type: MethodType, cover: np.ndarray, message: list[int]) -> np.ndarray:
    method = SteganographyMethodFactory.get(SAMPLE_RATE, method_type)
    return np.asarray(method.encode(cover.copy(), list(message)))


def _decode(method_type: MethodType, samples: np.ndarray, length: int) -> list[int]:
    method = SteganographyMethodFactory.get(SAMPLE_RATE, method_type)
    return [int(bit) for bit in method.decode(np.asarray(samples), length)]


@pytest.mark.parametrize("method_type", METHODS, ids=lambda m: m.name)
def test_payload_survives_when_nothing_is_done(
    method_type: MethodType, speech_cover: np.ndarray, message: list[int]
) -> None:
    """The control condition: without it, a zero BER under attack means nothing."""
    stego = _stego(method_type, speech_cover, message)
    assert bit_error_rate(message, _decode(method_type, stego, len(message))) == 0.0


@pytest.mark.parametrize("method_type", METHODS, ids=lambda m: m.name)
@pytest.mark.parametrize(
    "spec",
    [
        "awgn:snr_db=20",
        "gain:gain_db=-6",
        "low_pass:cutoff_hz=4000",
        "bit_depth:bits=8",
        "crop:fraction=0.01",
        "time_shift:shift_ms=10",
        "echo:delay_ms=25,attenuation=0.25",
        "resample:intermediate_hz=8000",
    ],
)
def test_attack_decode_cycle_produces_a_usable_measurement(
    method_type: MethodType, speech_cover: np.ndarray, message: list[int], spec: str
) -> None:
    """Every (method, attack) pair yields a BER in [0, 1] and full metadata."""
    stego = _stego(method_type, speech_cover, message)
    result = build(spec, sample_rate=SAMPLE_RATE).apply(stego, SAMPLE_RATE)

    decoded = _decode(method_type, result.audio, len(message))
    ber = bit_error_rate(message, decoded)

    assert 0.0 <= ber <= 1.0
    assert len(decoded) == len(message)
    assert result.metadata["parameters"]
    assert result.metadata["input_length"] == len(stego)


def test_a_destructive_attack_actually_destroys(
    speech_cover: np.ndarray, message: list[int]
) -> None:
    """An attack that removes the embedding domain must show up as damage.

    LSB embedding lives in the low bits, so requantising to 8 bits has to
    break it. If this passed with a low BER, the attack would not be doing
    anything and every robustness number in the benchmark would be suspect.
    """
    stego = _stego(MethodType.LSB_METHOD, speech_cover, message)
    attacked = build("bit_depth:bits=8").apply(stego, SAMPLE_RATE)

    ber = bit_error_rate(message, _decode(MethodType.LSB_METHOD, attacked.audio, len(message)))
    assert ber > 0.2


def test_attack_severity_is_monotone_in_damage(
    speech_cover: np.ndarray, message: list[int]
) -> None:
    """Heavier noise must not produce a cleaner extraction, on average.

    Checked on a spread-spectrum method, whose detector degrades gradually
    with noise rather than cliff-edging, so the ordering is meaningful.
    """
    stego = _stego(MethodType.IMPROVED_SPREAD_SPECTRUM_METHOD, speech_cover, message)

    errors = []
    for snr in (40, 20, 0, -10):
        attacked = build(f"awgn:snr_db={snr},seed=3").apply(stego, SAMPLE_RATE)
        decoded = _decode(MethodType.IMPROVED_SPREAD_SPECTRUM_METHOD, attacked.audio, len(message))
        errors.append(bit_error_rate(message, decoded))

    assert errors[0] <= errors[-1]
    assert errors[-1] > 0.0


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")
def test_codec_attack_runs_end_to_end(speech_cover: np.ndarray, message: list[int]) -> None:
    stego = _stego(MethodType.QIM_METHOD, speech_cover, message)
    attacked = build("mp3:bitrate_kbps=128").apply(stego, SAMPLE_RATE)

    decoded = _decode(MethodType.QIM_METHOD, attacked.audio, len(message))
    assert 0.0 <= bit_error_rate(message, decoded) <= 1.0
    assert attacked.metadata["compressed_bytes"] > 0


def test_workflow_records_attack_parameters_in_the_result_row(
    speech_cover: np.ndarray,
) -> None:
    """The parameters must reach the result row, or the run is not reproducible."""
    from taf.evaluation.workflow import _apply_attack

    samples, rate, metadata = _apply_attack(speech_cover, SAMPLE_RATE, "awgn:snr_db=25,seed=5")

    assert rate == SAMPLE_RATE
    assert len(samples) == len(speech_cover)
    assert metadata["attack"] == "awgn"
    assert metadata["parameters"]["snr_db"] == 25.0
    assert metadata["parameters"]["seed"] == 5


def test_identical_configurations_produce_identical_results(
    speech_cover: np.ndarray, message: list[int]
) -> None:
    """Same input, same configuration, same seed - same audio and same BER."""
    stego = _stego(MethodType.QIM_METHOD, speech_cover, message)

    first = build("awgn:snr_db=15,seed=99").apply(stego, SAMPLE_RATE)
    second = build("awgn:snr_db=15,seed=99").apply(stego, SAMPLE_RATE)

    np.testing.assert_array_equal(first.audio, second.audio)
    assert bit_error_rate(message, _decode(MethodType.QIM_METHOD, first.audio, len(message))) == (
        bit_error_rate(message, _decode(MethodType.QIM_METHOD, second.audio, len(message)))
    )
