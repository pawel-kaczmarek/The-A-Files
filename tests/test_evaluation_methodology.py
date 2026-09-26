"""Methodological guarantees of the evaluation engine (taf.evaluation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from taf.evaluation import (
    EvaluationConfig,
    EvaluationMessage,
    FailureKind,
    RandomMessageSpec,
    evaluate_files,
)
from taf.evaluation.workflow import _materialize_messages
from taf.models.Metric import Metric
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.WavFile import WavFile
from taf.models.types import MethodType

SAMPLE_RATE = 16000


def _wav(name: str, seconds: float = 0.5, seed: int = 0) -> WavFile:
    rng = np.random.default_rng(seed)
    samples = (0.1 * rng.standard_normal(int(SAMPLE_RATE * seconds))).astype(np.float64)
    return WavFile(samplerate=SAMPLE_RATE, samples=samples, path=Path(f"C:/data/{name}.wav"))


class _Passthrough(SteganographyMethod):
    """Embeds nothing; decodes zeros. Enough to observe the engine."""

    def encode(self, data, message):
        return np.asarray(data).copy()

    def decode(self, data_with_watermark, watermark_length):
        return [0] * watermark_length

    def type(self) -> str:
        return "Passthrough"


class _Remembering(SteganographyMethod):
    """Returns the message it saw in encode(): only works if state leaks."""

    def __init__(self) -> None:
        self.message = None

    def encode(self, data, message):
        self.message = list(message)
        return np.asarray(data).copy()

    def decode(self, data_with_watermark, watermark_length):
        return self.message if self.message is not None else [0] * watermark_length

    def type(self) -> str:
        return "Remembering"


class _BrokenDecoder(_Passthrough):
    def decode(self, data_with_watermark, watermark_length):
        raise RuntimeError("decoder exploded")

    def type(self) -> str:
        return "Broken decoder"


class _CountingMetric(Metric):
    higher_is_better = True

    def __init__(self) -> None:
        self.calls = 0

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        self.calls += 1
        return 1.0

    def name(self) -> str:
        return "Counting"


# ------------------------------------------------------------------ messages


def test_messages_of_different_lengths_are_independent_draws():
    messages = _materialize_messages(
        EvaluationConfig(
            random_messages=(RandomMessageSpec(length=16), RandomMessageSpec(length=32)),
            random_seed=5,
        )
    )
    short, long = messages[0].bits, messages[1].bits
    assert long[:16] != short, "the 16-bit message must not be a prefix of the 32-bit one"


def test_adding_a_payload_length_does_not_change_the_other_messages():
    alone = _materialize_messages(
        EvaluationConfig(random_messages=(RandomMessageSpec(length=16, count=2),), random_seed=5)
    )
    together = _materialize_messages(
        EvaluationConfig(
            random_messages=(RandomMessageSpec(length=8), RandomMessageSpec(length=16, count=2)),
            random_seed=5,
        )
    )
    assert [m.bits for m in alone] == [m.bits for m in together if m.length == 16]


# ---------------------------------------------------------- attack seeding


def _attack_seed(row) -> int:
    return row.attack_parameters["parameters"]["seed"]


def test_attack_noise_is_shared_by_methods_and_resampled_per_repetition_and_file():
    result = evaluate_files(
        [_wav("a", seed=1), _wav("b", seed=2)],
        EvaluationConfig(
            methods=[MethodType.LSB_METHOD, lambda sr: _Passthrough()],
            metrics=[],
            random_messages=(RandomMessageSpec(length=8, count=2),),
            random_seed=11,
            attacks=["awgn:snr_db=30"],
        ),
    )
    attacked = [row for row in result.rows if row.attack is not None]
    assert attacked and all(row.error is None for row in attacked)

    seeds: dict[tuple[str, int], set[int]] = {}
    for row in attacked:
        seeds.setdefault((row.input_path.name, row.repetition), set()).add(_attack_seed(row))

    # Common random numbers: every method met the same noise in one trial.
    assert all(len(values) == 1 for values in seeds.values())
    # Repetitions and files sample the channel independently.
    distinct = {next(iter(values)) for values in seeds.values()}
    assert len(distinct) == len(seeds) == 4


def test_an_explicit_attack_seed_is_respected():
    result = evaluate_files(
        [_wav("a")],
        EvaluationConfig(
            methods=[lambda sr: _Passthrough()],
            metrics=[],
            random_messages=(RandomMessageSpec(length=8, count=2),),
            random_seed=11,
            attacks=["awgn:snr_db=30,seed=7"],
        ),
    )
    assert {_attack_seed(row) for row in result.rows if row.attack} == {7}


def test_attack_realisations_are_reproducible_from_the_seed():
    def seeds(seed: int) -> list[int]:
        result = evaluate_files(
            [_wav("a")],
            EvaluationConfig(
                methods=[lambda sr: _Passthrough()],
                metrics=[],
                random_messages=(RandomMessageSpec(length=8, count=2),),
                random_seed=seed,
                attacks=["pink_noise:snr_db=20"],
            ),
        )
        return sorted(_attack_seed(row) for row in result.rows if row.attack)

    assert seeds(3) == seeds(3)
    assert seeds(3) != seeds(4)


# --------------------------------------------------------------- failures


def test_over_capacity_is_recorded_as_such_not_as_a_crash():
    result = evaluate_files(
        [_wav("tiny", seconds=0.001)],  # 16 samples
        EvaluationConfig(
            methods=[MethodType.LSB_METHOD],
            metrics=[],
            messages=(EvaluationMessage(name="long", bits=(1,) * 64, index=0),),
        ),
    )
    (row,) = result.rows
    assert row.failure_kind == FailureKind.OVER_CAPACITY
    assert row.decoded_message is None


def test_decode_failure_is_classified_and_keeps_the_embedding_metrics():
    counting = _CountingMetric()
    result = evaluate_files(
        [_wav("a")],
        EvaluationConfig(
            methods=[lambda sr: _BrokenDecoder()],
            metrics=[lambda: counting],
            random_messages=(RandomMessageSpec(length=8),),
            random_seed=1,
        ),
    )
    (row,) = result.rows
    assert row.failure_kind == FailureKind.DECODE_ERROR
    # Imperceptibility was measurable although extraction failed.
    assert row.metrics == {"Counting": 1.0}


def test_extraction_uses_a_fresh_decoder():
    result = evaluate_files(
        [_wav("a")],
        EvaluationConfig(
            methods=[lambda sr: _Remembering()],
            metrics=[],
            messages=(EvaluationMessage(name="ones", bits=(1,) * 8, index=0),),
        ),
    )
    (row,) = result.rows
    assert row.success is False, "state kept by encode() must not reach decode()"


def test_embedding_metrics_are_computed_once_per_encoded_signal():
    counting = _CountingMetric()
    evaluate_files(
        [_wav("a")],
        EvaluationConfig(
            methods=[lambda sr: _Passthrough()],
            metrics=[lambda: counting],
            random_messages=(RandomMessageSpec(length=8),),
            random_seed=1,
            attacks=["gain:gain_db=-3", "awgn:snr_db=30"],
        ),
    )
    # One cover-vs-stego evaluation, plus one stego-vs-attacked per attack.
    assert counting.calls == 1 + 2
