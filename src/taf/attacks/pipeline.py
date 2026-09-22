"""Composition of several attacks into one reproducible channel.

Real distribution chains are not single operations. A file uploaded to a
platform is transcoded, possibly resampled, normalised and transcoded again;
a clip captured from a speaker is filtered, reverberated, noised and
re-clocked. Methods can survive each stage alone and fail the sequence,
because the damage compounds and, more importantly, because the stages
interact: a resampler after a codec moves the codec's quantisation noise into
different bins.

A pipeline applies its stages in order and keeps the metadata of each one, so
a result row still records exactly what happened rather than just a label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from taf.attacks.base import Attack, AttackCategory, AttackError, AttackResult


@dataclass(frozen=True)
class AttackPipeline(Attack):
    """Apply several attacks in sequence.

    The pipeline is itself an :class:`Attack`, so anything that accepts an
    attack accepts a chain, and a chain can be nested in another chain.
    """

    stages: tuple[Attack, ...] = ()
    label: str = "pipeline"

    name = "pipeline"
    category = AttackCategory.PIPELINE

    def __post_init__(self) -> None:
        if not self.stages:
            raise AttackError("a pipeline needs at least one stage")
        for stage in self.stages:
            if not isinstance(stage, Attack):
                raise AttackError(f"pipeline stages must be attacks, got {stage!r}")

    @property
    def changes_length_or_rate(self) -> bool:  # type: ignore[override]
        return any(stage.changes_length_or_rate for stage in self.stages)

    def parameters(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "stages": [
                {"attack": stage.name, "parameters": stage.parameters()} for stage in self.stages
            ],
        }

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        signal = audio
        rate = sample_rate
        records: list[dict[str, Any]] = []

        for stage in self.stages:
            # Each stage is run through its own apply(), so every stage's
            # validation, dtype handling and metadata are exactly what they
            # would be if it were applied on its own.
            result = stage.apply(signal, rate)
            signal = np.asarray(result.audio, dtype=np.float64)
            rate = result.sample_rate
            records.append(result.metadata)

        return signal, rate, {"label": self.label, "stage_metadata": records}


def chain(*stages: Attack, label: str = "pipeline") -> AttackPipeline:
    """Build a pipeline from attacks given positionally."""
    return AttackPipeline(stages=tuple(stages), label=label)


def apply_sequence(
    audio: np.ndarray, sample_rate: int, stages: Sequence[Attack], label: str = "pipeline"
) -> AttackResult:
    """Convenience wrapper: build a pipeline and apply it immediately."""
    return chain(*stages, label=label).apply(audio, sample_rate)


__all__ = ["AttackPipeline", "apply_sequence", "chain"]
