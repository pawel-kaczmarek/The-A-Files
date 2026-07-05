"""Method comparison: weighted overall ranking under identical conditions."""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    by_method,
    group_by,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.perceptual_quality import quality_ranking
from taf.experiments.schema import ExperimentConfig, ExperimentType

DEFAULT_WEIGHTS = {"quality": 0.30, "robustness": 0.30, "accuracy": 0.30, "speed": 0.10}


def weights_from(config: ExperimentConfig) -> dict[str, float]:
    options = (config.advanced_options or {}).get("weights", {})
    weights = {key: float(options.get(key, default)) for key, default in DEFAULT_WEIGHTS.items()}
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in weights.items()}


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
    weights = weights_from(config)
    methods = by_method(rows)
    names = [entry["method"] for entry in methods]

    quality_scores = {
        entry["method"]: entry["quality_score"] for entry in quality_ranking(methods)
    } or {name: None for name in names}

    attacked = [row for row in rows if row.attack is not None]
    attacked_stats = {
        method: group_stats(group) for method, group in group_by(attacked, lambda r: r.method).items()
    }
    robustness_scores = {
        name: (attacked_stats.get(name) or {}).get("avg_bit_accuracy") for name in names
    }

    accuracy_scores = {
        entry["method"]: entry["avg_bit_accuracy"] for entry in methods
    }
    time_totals = {
        entry["method"]: (
            None
            if entry["avg_encode_time_seconds"] is None and entry["avg_decode_time_seconds"] is None
            else (entry["avg_encode_time_seconds"] or 0.0) + (entry["avg_decode_time_seconds"] or 0.0)
        )
        for entry in methods
    }
    speed_scores = _minmax(time_totals, invert=True)  # faster = better

    table: list[dict[str, Any]] = []
    for entry in methods:
        name = entry["method"]
        components = {
            "quality": quality_scores.get(name),
            "robustness": robustness_scores.get(name),
            "accuracy": accuracy_scores.get(name),
            "speed": speed_scores.get(name),
        }
        weighted = [
            (weights[key], value) for key, value in components.items() if value is not None
        ]
        weight_sum = sum(weight for weight, _ in weighted)
        overall = (
            sum(weight * value for weight, value in weighted) / weight_sum if weight_sum > 0 else None
        )
        table.append(
            {
                "method": name,
                "overall_score": overall,
                **{f"{key}_score": value for key, value in components.items()},
                "avg_ber": entry["avg_ber"],
                "avg_bit_accuracy": entry["avg_bit_accuracy"],
                "avg_encode_time_seconds": entry["avg_encode_time_seconds"],
                "avg_decode_time_seconds": entry["avg_decode_time_seconds"],
            }
        )
    table.sort(key=lambda e: (e["overall_score"] is None, -(e["overall_score"] or 0.0)))
    for rank, entry in enumerate(table, start=1):
        entry["rank"] = rank

    return {
        "overall": group_stats(rows),
        "weights": weights,
        "comparison": table,
        "best_method": table[0]["method"] if table else None,
        "by_method": methods,
    }


def _validate(config: ExperimentConfig) -> list[str]:
    if len(config.methods) < 2:
        return ["Method comparison needs at least two methods."]
    return []


SCENARIO = Scenario(
    experiment_type=ExperimentType.METHOD_COMPARISON,
    title="Method Comparison",
    description=(
        "Compare methods under identical conditions and rank them with a weighted score "
        "over quality, robustness, accuracy and speed."
    ),
    validate=_validate,
    summarize=_summarize,
)
