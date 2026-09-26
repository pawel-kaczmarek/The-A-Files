"""Analyses shared by several scenarios."""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.analysis import paired_comparison, pareto_front, trial_ber
from taf.experiments.results import (
    ExperimentResultRow,
    attacked_rows,
    baseline_rows,
    group_by,
    group_stats,
    metric_direction,
)


def ber_comparison(rows: Sequence[ExperimentResultRow]) -> dict[str, Any]:
    """Paired comparison of methods on BER, failures scored at chance level."""
    return paired_comparison(
        rows, lambda row: trial_ber(row, impute_failures=True), higher_is_better=False
    )


def ranked_metrics(rows: Sequence[ExperimentResultRow]) -> dict[str, bool]:
    """Metric labels present in ``rows`` whose direction is declared."""
    names = sorted({name for row in rows for name in row.metrics})
    return {
        name: direction
        for name in names
        if (direction := metric_direction(name)) is not None
    }


def metric_comparisons(rows: Sequence[ExperimentResultRow]) -> dict[str, Any]:
    """One paired comparison per ranked metric (cover vs stego)."""
    return {
        name: paired_comparison(
            rows, lambda row, n=name: row.metrics.get(n), higher_is_better=direction
        )
        for name, direction in ranked_metrics(rows).items()
    }


def quality_ranking(rows: Sequence[ExperimentResultRow]) -> list[dict[str, Any]]:
    """Methods ordered by their mean rank over the ranked metrics (1 = best).

    Ranks are invariant to the scale of each metric, so a metric in decibels
    and one on a 1-5 opinion scale weigh the same, and no metric's direction
    is guessed. Metrics without a declared direction are left out.
    """
    directions = ranked_metrics(rows)
    methods = group_by(rows, lambda row: row.method)
    means: dict[str, dict[str, float]] = {}
    for method, group in methods.items():
        stats = group_stats(group)["avg_metrics"]
        means[method] = {name: stats[name] for name in directions if name in stats}

    rank_lists: dict[str, list[float]] = {method: [] for method in methods}
    for name, higher_is_better in directions.items():
        scored = [(method, values[name]) for method, values in means.items() if name in values]
        if len(scored) < 2:
            continue
        ordered = sorted(scored, key=lambda item: item[1], reverse=higher_is_better)
        position = 0
        while position < len(ordered):
            tied = [
                item for item in ordered[position:] if item[1] == ordered[position][1]
            ]
            rank = position + (len(tied) + 1) / 2
            for method, _ in tied:
                rank_lists[method].append(rank)
            position += len(tied)

    ranking = [
        {
            "method": method,
            "mean_rank": sum(ranks) / len(ranks) if ranks else None,
            "metrics_ranked": len(ranks),
            "avg_metrics": group_stats(methods[method])["avg_metrics"],
        }
        for method, ranks in rank_lists.items()
    ]
    ranking.sort(key=lambda entry: (entry["mean_rank"] is None, entry["mean_rank"] or 0.0))
    return ranking


def method_objectives(
    rows: Sequence[ExperimentResultRow], include_speed: bool = False
) -> tuple[dict[str, dict[str, float | None]], dict[str, bool]]:
    """Per-method objective values and directions for a Pareto analysis."""
    baseline = baseline_rows(rows)
    attacked = attacked_rows(rows)
    directions: dict[str, bool] = {"bit_accuracy": True}
    if attacked:
        directions["robustness"] = True
    metrics = ranked_metrics(rows)
    directions.update({f"metric: {name}": higher for name, higher in metrics.items()})
    if include_speed:
        directions["time_seconds"] = False

    scores: dict[str, dict[str, float | None]] = {}
    for method, group in group_by(rows, lambda row: row.method).items():
        base = group_stats([row for row in baseline if row.method == method])
        entry: dict[str, float | None] = {"bit_accuracy": base["avg_bit_accuracy_imputed"]}
        if attacked:
            hit = group_stats([row for row in attacked if row.method == method])
            entry["robustness"] = hit["avg_bit_accuracy_imputed"]
        averages = group_stats(group)["avg_metrics"]
        for name in metrics:
            entry[f"metric: {name}"] = averages.get(name)
        if include_speed:
            stats = group_stats(group)
            encode, decode = stats["avg_encode_time_seconds"], stats["avg_decode_time_seconds"]
            entry["time_seconds"] = (
                None if encode is None and decode is None else (encode or 0.0) + (decode or 0.0)
            )
        scores[method] = entry
    return scores, directions


def method_pareto(rows: Sequence[ExperimentResultRow], include_speed: bool = False) -> dict[str, Any]:
    scores, directions = method_objectives(rows, include_speed=include_speed)
    return pareto_front(scores, directions)


def ranking_from(comparison: dict[str, Any], fallback: list[str]) -> list[str]:
    """Methods by mean rank when a paired comparison exists, else ``fallback``."""
    if comparison.get("available"):
        return list(comparison["mean_ranks"])
    return fallback


__all__ = [
    "ber_comparison",
    "method_objectives",
    "method_pareto",
    "metric_comparisons",
    "quality_ranking",
    "ranked_metrics",
    "ranking_from",
]
