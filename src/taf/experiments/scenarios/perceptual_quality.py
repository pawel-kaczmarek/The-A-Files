"""Perceptual quality: how much does each method degrade the audio?"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    by_method,
    by_method_payload,
    group_stats,
    metric_is_lower_better,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.schema import ExperimentConfig, ExperimentType


def quality_ranking(method_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank methods by mean min-max-normalized metric score (0..1, higher = better).

    Metric direction uses a name heuristic (``metric_is_lower_better``); the
    detailed per-metric averages remain available for exact interpretation.
    """
    metric_names = sorted({name for entry in method_stats for name in entry["avg_metrics"]})
    if not metric_names:
        return []
    ranking = []
    for entry in method_stats:
        scores: list[float] = []
        for name in metric_names:
            values = [
                other["avg_metrics"][name]
                for other in method_stats
                if name in other["avg_metrics"]
            ]
            value = entry["avg_metrics"].get(name)
            if value is None or not values:
                continue
            low, high = min(values), max(values)
            normalized = 0.5 if high == low else (value - low) / (high - low)
            if metric_is_lower_better(name):
                normalized = 1.0 - normalized
            scores.append(normalized)
        ranking.append(
            {
                "method": entry["method"],
                "quality_score": sum(scores) / len(scores) if scores else None,
                "avg_metrics": entry["avg_metrics"],
            }
        )
    ranking.sort(key=lambda e: (e["quality_score"] is None, -(e["quality_score"] or 0.0)))
    return ranking


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    methods = by_method(rows)
    return {
        "overall": group_stats(rows),
        "by_method": methods,
        "by_method_payload": by_method_payload(rows),
        "quality_ranking": quality_ranking(methods),
    }


SCENARIO = Scenario(
    experiment_type=ExperimentType.PERCEPTUAL_QUALITY,
    title="Perceptual Quality",
    description=(
        "Quantify how much audio quality is degraded by hiding information: original vs "
        "watermarked signal compared with the selected quality metrics."
    ),
    requires_metrics=True,
    summarize=_summarize,
)
