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
from taf.experiments.scenarios.detectability import SCENARIO as _detectability
from taf.experiments.scenarios.embedding_capacity import SCENARIO as _embedding_capacity
from taf.experiments.scenarios.evaluation import evaluation_statistics, evaluation_summary
from taf.experiments.scenarios.method_comparison import SCENARIO as _method_comparison
from taf.experiments.scenarios.perceptual_quality import SCENARIO as _perceptual_quality
from taf.experiments.scenarios.research_experiment import SCENARIO as _research_experiment
from taf.experiments.scenarios.robustness_curve import SCENARIO as _robustness_curve
from taf.experiments.scenarios.tradeoff_curve import SCENARIO as _tradeoff_curve
from taf.experiments.schema import ExperimentConfig, ExperimentType

SCENARIOS: dict[ExperimentType, Scenario] = {
    scenario.experiment_type: scenario
    # Ordered as a catalogue: one design per property first, then the
    # designs that weigh several properties.
    for scenario in (
        _perceptual_quality,
        _attack_robustness,
        _robustness_curve,
        _embedding_capacity,
        _detectability,
        _tradeoff_curve,
        _method_comparison,
        _dataset_benchmark,
        _research_experiment,
    )
}


def get_scenario(experiment_type: ExperimentType | str) -> Scenario:
    return SCENARIOS[ExperimentType(experiment_type)]


def describe_designs() -> list[dict[str, Any]]:
    """Every experimental design with its property, factors, measures and analyses."""
    return [scenario.describe() for scenario in SCENARIOS.values()]


def validate_for_scenario(config: ExperimentConfig) -> list[str]:
    scenario = get_scenario(config.experiment_type)
    problems: list[str] = []
    if scenario.requires_metrics and not config.metrics:
        problems.append(f"{scenario.title} requires at least one metric.")
    if len(config.resolved_methods()) < scenario.min_methods:
        problems.append(f"{scenario.title} requires at least {scenario.min_methods} methods.")
    if scenario.requires_attacks and not (config.attacks or config.attack_preset or config.attack_sweep):
        problems.append(f"{scenario.title} requires at least one attack.")
    return problems + scenario.validate(config)


def summarize_for_scenario(
    rows: Sequence[ExperimentResultRow], config: ExperimentConfig
) -> dict[str, Any]:
    """The design's own analysis, plus the shared evaluation block.

    Comparisons the evaluation adds (per attack, per quality metric) join the
    design's ``statistics`` only where the design has not computed them.
    """
    summary = get_scenario(config.experiment_type).summarize(rows, config)
    if rows:
        statistics = summary.setdefault("statistics", {})
        for key, value in evaluation_statistics(rows, config).items():
            # A sweep design already compares methods at every swept attack.
            if key == "per_attack" and "per_value" in statistics:
                continue
            statistics.setdefault(key, value)
        summary["evaluation"] = evaluation_summary(rows, config, statistics)
    return summary


__all__ = [
    "SCENARIOS",
    "Scenario",
    "describe_designs",
    "get_scenario",
    "summarize_for_scenario",
    "validate_for_scenario",
]
