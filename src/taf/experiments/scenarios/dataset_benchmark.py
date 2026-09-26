"""Dataset benchmark: the broad, general-purpose scenario."""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    attacked_rows,
    baseline_rows,
    by_method,
    by_method_attack,
    by_method_payload,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.common import ber_comparison, method_pareto, ranking_from
from taf.experiments.schema import ExperimentConfig, ExperimentType


def benchmark_summary(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    """Summary shared by the benchmark and the free-form research scenario.

    Rankings use the no-attack rows only: mixing in attacked rows would make a
    method's rank depend on which attacks happened to be selected. Attacked
    rows are summarised and compared on their own.
    """
    baseline = baseline_rows(rows)
    base_methods = by_method(baseline)
    fallback = [
        entry["method"]
        for entry in sorted(
            base_methods,
            key=lambda e: (
                e["avg_bit_accuracy_imputed"] is None,
                -(e["avg_bit_accuracy_imputed"] or 0.0),
            ),
        )
    ]
    statistics: dict[str, Any] = {"ber_baseline": ber_comparison(baseline)}
    summary: dict[str, Any] = {
        "overall": group_stats(rows),
        "baseline": group_stats(baseline),
        "by_method": by_method(rows),
        "method_ranking": ranking_from(statistics["ber_baseline"], fallback),
        "by_method_payload": by_method_payload(rows),
        "statistics": statistics,
        "pareto": method_pareto(rows),
    }
    if config.attacks or config.attack_preset:
        attacked = attacked_rows(rows)
        summary["attacked"] = group_stats(attacked)
        summary["by_method_attack"] = by_method_attack(rows)
        statistics["ber_attacked"] = ber_comparison(attacked)
    return summary


SCENARIO = Scenario(
    experiment_type=ExperimentType.DATASET_BENCHMARK,
    title="Dataset Benchmark",
    description=(
        "Run selected steganography methods over a dataset for chosen payload lengths, "
        "metrics and optional attacks — the general benchmark scenario."
    ),
    property="multi_criteria",
    factors=("method", "payload_length", "attack"),
    measures=("ber", "quality_metrics", "completion_rate"),
    analyses=("cluster_bootstrap_ci", "friedman_holm_wilcoxon", "pareto_front"),
    summarize=benchmark_summary,
)
