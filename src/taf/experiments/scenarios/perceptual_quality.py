"""Perceptual quality: how much does each method degrade the audio?"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    baseline_rows,
    by_method,
    by_method_payload,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.common import (
    metric_comparisons,
    method_pareto,
    quality_ranking,
    ranked_metrics,
)
from taf.experiments.schema import ExperimentConfig, ExperimentType


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    # Cover-vs-stego metrics do not depend on the attack, so the no-attack
    # rows carry all of them once per trial.
    baseline = baseline_rows(rows)
    all_metrics = {name for row in baseline for name in row.metrics}
    return {
        "overall": group_stats(rows),
        "by_method": by_method(baseline),
        "by_method_payload": by_method_payload(baseline),
        "quality_ranking": quality_ranking(baseline),
        # Reported but not ranked: metrics whose direction is not declared,
        # and reference entries such as the SRMR of the cover.
        "unranked_metrics": sorted(all_metrics - set(ranked_metrics(baseline))),
        "statistics": {"metrics": metric_comparisons(baseline)},
        "pareto": method_pareto(baseline),
    }


SCENARIO = Scenario(
    experiment_type=ExperimentType.PERCEPTUAL_QUALITY,
    title="Perceptual Quality",
    description=(
        "Quantify how much audio quality is degraded by hiding information: original vs "
        "watermarked signal compared with the selected quality metrics."
    ),
    property="imperceptibility",
    factors=("method", "payload_length"),
    measures=("quality_metrics", "ber"),
    analyses=("cluster_bootstrap_ci", "rank_aggregation", "friedman_holm_wilcoxon", "pareto_front"),
    requires_metrics=True,
    summarize=_summarize,
)
