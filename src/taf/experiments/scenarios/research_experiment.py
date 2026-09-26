"""Research experiment: fully configurable custom scenario."""

from __future__ import annotations

from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.dataset_benchmark import benchmark_summary
from taf.experiments.schema import ExperimentType

SCENARIO = Scenario(
    experiment_type=ExperimentType.RESEARCH_EXPERIMENT,
    title="Research Experiment",
    description=(
        "Full control over the shared experiment configuration for custom research "
        "scenarios: any combination of datasets, methods, metrics, attacks, payloads "
        "and output options."
    ),
    property="multi_criteria",
    factors=("method", "payload_length", "attack", "repetition"),
    measures=("ber", "quality_metrics", "attack_metrics", "time"),
    analyses=("cluster_bootstrap_ci", "friedman_holm_wilcoxon", "pareto_front"),
    summarize=benchmark_summary,
)
