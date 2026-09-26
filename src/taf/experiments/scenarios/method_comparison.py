"""Method comparison: methods measured under identical conditions.

The primary result is the Pareto front over the measured criteria plus a
paired significance test per criterion. A single weighted score is only
computed when the configuration supplies weights: it depends on those
weights and, through the normalisation, on which other methods are in the
comparison, so it is a statement of preference rather than a measurement.
"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    attacked_rows,
    baseline_rows,
    by_method,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.common import (
    ber_comparison,
    method_objectives,
    metric_comparisons,
    quality_ranking,
)
from taf.experiments.analysis import pareto_front
from taf.experiments.schema import ExperimentConfig, ExperimentType

WEIGHT_KEYS = ("quality", "robustness", "accuracy", "speed")


def weights_from(config: ExperimentConfig) -> dict[str, float] | None:
    """Explicit weights from ``advanced_options.weights``, normalised; else ``None``."""
    options = (config.advanced_options or {}).get("weights")
    if not options:
        return None
    weights = {key: max(0.0, float(options.get(key, 0.0))) for key in WEIGHT_KEYS}
    total = sum(weights.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in weights.items()}


def timing_is_reliable(config: ExperimentConfig) -> bool:
    """Timings are comparable only when trials did not run concurrently."""
    return config.max_workers == 1


def _minmax(values: dict[str, float | None], invert: bool = False) -> dict[str, float | None]:
    present = [v for v in values.values() if v is not None]
    if not present:
        return {k: None for k in values}
    low, high = min(present), max(present)
    normalized: dict[str, float | None] = {}
    for key, value in values.items():
        if value is None:
            normalized[key] = None
            continue
        score = 0.5 if high == low else (value - low) / (high - low)
        normalized[key] = 1.0 - score if invert else score
    return normalized


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    baseline = baseline_rows(rows)
    attacked = attacked_rows(rows)
    reliable_timing = timing_is_reliable(config)

    scores, directions = method_objectives(rows, include_speed=reliable_timing)
    pareto = pareto_front(scores, directions)
    quality = {entry["method"]: entry for entry in quality_ranking(baseline)}

    statistics: dict[str, Any] = {
        "ber_baseline": ber_comparison(baseline),
        "metrics": metric_comparisons(baseline),
    }
    if attacked:
        statistics["ber_attacked"] = ber_comparison(attacked)

    table: list[dict[str, Any]] = []
    for method, values in scores.items():
        stats = group_stats([row for row in rows if row.method == method])
        table.append(
            {
                "method": method,
                "pareto_optimal": method in pareto["front"],
                "dominated_by": ", ".join(pareto["dominated_by"].get(method, [])) or None,
                "accuracy_score": values.get("bit_accuracy"),
                "robustness_score": values.get("robustness"),
                "quality_mean_rank": (quality.get(method) or {}).get("mean_rank"),
                "avg_ber": stats["avg_ber"],
                "completion_rate": stats["completion_rate"],
                "avg_encode_time_seconds": stats["avg_encode_time_seconds"],
                "avg_decode_time_seconds": stats["avg_decode_time_seconds"],
            }
        )

    weights = weights_from(config)
    if weights is not None:
        quality_scores = _minmax(
            {entry["method"]: entry["quality_mean_rank"] for entry in table}, invert=True
        )
        speed_scores = _minmax(
            {
                entry["method"]: (
                    None
                    if entry["avg_encode_time_seconds"] is None and entry["avg_decode_time_seconds"] is None
                    else (entry["avg_encode_time_seconds"] or 0.0) + (entry["avg_decode_time_seconds"] or 0.0)
                )
                for entry in table
            },
            invert=True,
        )
        for entry in table:
            components = {
                "quality": quality_scores.get(entry["method"]),
                "robustness": entry["robustness_score"],
                "accuracy": entry["accuracy_score"],
                "speed": speed_scores.get(entry["method"]) if reliable_timing else None,
            }
            weighted = [(weights[key], value) for key, value in components.items() if value is not None]
            weight_sum = sum(weight for weight, _ in weighted)
            entry["weighted_score"] = (
                sum(weight * value for weight, value in weighted) / weight_sum if weight_sum > 0 else None
            )

    # Pareto-optimal methods first; within each group by accuracy.
    table.sort(
        key=lambda e: (
            not e["pareto_optimal"],
            e["accuracy_score"] is None,
            -(e["accuracy_score"] or 0.0),
        )
    )
    for rank, entry in enumerate(table, start=1):
        entry["rank"] = rank

    front = pareto["front"]
    return {
        "overall": group_stats(rows),
        "comparison": table,
        "pareto": pareto,
        # A single best method exists only when one method dominates all others.
        "best_method": front[0] if len(front) == 1 else None,
        "statistics": statistics,
        "weights": weights,
        "weighting": "user-supplied, subjective" if weights is not None else None,
        "timing_reliable": reliable_timing,
        "by_method": by_method(rows),
    }


SCENARIO = Scenario(
    experiment_type=ExperimentType.METHOD_COMPARISON,
    title="Method Comparison",
    description=(
        "Compare methods under identical conditions: Pareto front over accuracy, "
        "robustness and quality, with a paired significance test per criterion."
    ),
    property="multi_criteria",
    factors=("method", "attack"),
    measures=("ber", "quality_metrics", "ber_under_attack", "time"),
    analyses=("pareto_front", "friedman_holm_wilcoxon", "cluster_bootstrap_ci"),
    min_methods=2,
    summarize=_summarize,
)
