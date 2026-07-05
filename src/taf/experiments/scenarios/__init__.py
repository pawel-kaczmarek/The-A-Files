"""Experiment scenarios: per-type validation rules and summary analytics.

All scenarios share the same execution pipeline (``taf.experiments.runner``);
what differs is which inputs are mandatory and how results are summarized.
"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import ExperimentResultRow
from taf.experiments.scenarios.attack_robustness import SCENARIO as _attack_robustness
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.dataset_benchmark import SCENARIO as _dataset_benchmark
from taf.experiments.scenarios.embedding_capacity import SCENARIO as _embedding_capacity
from taf.experiments.scenarios.method_comparison import SCENARIO as _method_comparison
from taf.experiments.scenarios.perceptual_quality import SCENARIO as _perceptual_quality
from taf.experiments.scenarios.research_experiment import SCENARIO as _research_experiment
from taf.experiments.schema import ExperimentConfig, ExperimentType

SCENARIOS: dict[ExperimentType, Scenario] = {
    scenario.experiment_type: scenario
    for scenario in (
        _dataset_benchmark,
        _attack_robustness,
        _perceptual_quality,
        _embedding_capacity,
        _method_comparison,
        _research_experiment,
    )
}


def get_scenario(experiment_type: ExperimentType | str) -> Scenario:
    return SCENARIOS[ExperimentType(experiment_type)]


def validate_for_scenario(config: ExperimentConfig) -> list[str]:
    scenario = get_scenario(config.experiment_type)
    problems: list[str] = []
    if scenario.requires_metrics and not config.metrics:
        problems.append(f"{scenario.title} requires at least one metric.")
    if scenario.requires_attacks and not config.attacks:
        problems.append(f"{scenario.title} requires at least one attack.")
    return problems + scenario.validate(config)


def summarize_for_scenario(
    rows: Sequence[ExperimentResultRow], config: ExperimentConfig
) -> dict[str, Any]:
    return get_scenario(config.experiment_type).summarize(rows, config)


__all__ = [
    "SCENARIOS",
    "Scenario",
    "get_scenario",
    "summarize_for_scenario",
    "validate_for_scenario",
]
