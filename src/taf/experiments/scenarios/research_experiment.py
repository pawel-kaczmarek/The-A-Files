"""Research experiment: fully configurable custom scenario."""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    by_method,
    by_method_attack,
    by_method_payload,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.schema import ExperimentConfig, ExperimentType


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall": group_stats(rows),
        "by_method": by_method(rows),
        "by_method_payload": by_method_payload(rows),
    }
    if config.attacks:
        summary["by_method_attack"] = by_method_attack(rows)
    return summary


SCENARIO = Scenario(
    experiment_type=ExperimentType.RESEARCH_EXPERIMENT,
    title="Research Experiment",
    description=(
        "Full control over the shared experiment configuration for custom research "
        "scenarios: any combination of datasets, methods, metrics, attacks, payloads "
        "and output options."
    ),
    summarize=_summarize,
)
