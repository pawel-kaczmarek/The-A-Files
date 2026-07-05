"""Scenario contract shared by all experiment types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from taf.experiments.results import ExperimentResultRow
from taf.experiments.schema import ExperimentConfig, ExperimentType


@dataclass(frozen=True)
class Scenario:
    experiment_type: ExperimentType
    title: str
    description: str
    requires_metrics: bool = False
    requires_attacks: bool = False
    default_payload_lengths: tuple[int, ...] = (16,)
    validate: Callable[[ExperimentConfig], list[str]] = field(default=lambda config: [])
    summarize: Callable[[Sequence[ExperimentResultRow], ExperimentConfig], dict[str, Any]] = field(
        default=lambda rows, config: {}
    )


__all__ = ["Scenario"]
