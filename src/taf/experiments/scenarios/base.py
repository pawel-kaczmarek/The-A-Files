"""Scenario contract shared by all experiment types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from taf.experiments.results import ExperimentResultRow
from taf.experiments.schema import ExperimentConfig, ExperimentType

#: The four properties an information-hiding method is evaluated on, plus
#: designs that weigh several of them at once.
PROPERTIES = ("imperceptibility", "robustness", "capacity", "security", "multi_criteria")


@dataclass(frozen=True)
class Scenario:
    experiment_type: ExperimentType
    title: str
    description: str
    #: The property the design measures (one of ``PROPERTIES``).
    property: str = "multi_criteria"
    #: Independent variables the design varies.
    factors: tuple[str, ...] = ("method", "payload_length")
    #: Dependent variables it measures.
    measures: tuple[str, ...] = ("ber",)
    #: Analyses applied to the results.
    analyses: tuple[str, ...] = ()
    requires_metrics: bool = False
    requires_attacks: bool = False
    #: ``"attack"`` or ``"method"`` when the design sweeps a parameter.
    requires_sweep: str | None = None
    min_methods: int = 1
    default_payload_lengths: tuple[int, ...] = (16,)
    validate: Callable[[ExperimentConfig], list[str]] = field(default=lambda config: [])
    summarize: Callable[[Sequence[ExperimentResultRow], ExperimentConfig], dict[str, Any]] = field(
        default=lambda rows, config: {}
    )

    def describe(self) -> dict[str, Any]:
        """Structure of the design, for catalogues and user interfaces."""
        return {
            "type": self.experiment_type.value,
            "title": self.title,
            "description": self.description,
            "property": self.property,
            "factors": list(self.factors),
            "measures": list(self.measures),
            "analyses": list(self.analyses),
            "requires_metrics": self.requires_metrics,
            "requires_attacks": self.requires_attacks,
            "requires_sweep": self.requires_sweep,
            "min_methods": self.min_methods,
            "default_payload_lengths": list(self.default_payload_lengths),
        }


__all__ = ["PROPERTIES", "Scenario"]
