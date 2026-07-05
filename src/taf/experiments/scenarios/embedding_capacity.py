"""Embedding capacity: practical maximum payload per method."""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    by_method_payload,
    group_by,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.schema import ExperimentConfig, ExperimentType

DEFAULT_MIN_BIT_ACCURACY = 0.95
DEFAULT_MAX_BER = 0.05
PAYLOAD_PRESETS = (4, 8, 16, 32, 64, 120)


def thresholds_from(config: ExperimentConfig) -> tuple[float, float]:
    options = config.advanced_options or {}
    return (
        float(options.get("min_bit_accuracy", DEFAULT_MIN_BIT_ACCURACY)),
        float(options.get("max_ber", DEFAULT_MAX_BER)),
    )


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    min_accuracy, max_ber = thresholds_from(config)
    cells = by_method_payload(rows)
    for cell in cells:
        accuracy = cell["avg_bit_accuracy"]
        ber = cell["avg_ber"]
        cell["passes"] = (
            accuracy is not None
            and ber is not None
            and accuracy >= min_accuracy
            and ber <= max_ber
        )

    capacity: list[dict[str, Any]] = []
    for method, method_cells in sorted(
        group_by(cells, lambda c: c["method"]).items()  # type: ignore[arg-type]
    ):
        ordered = sorted(method_cells, key=lambda c: c["payload_length"])
        passing = [c["payload_length"] for c in ordered if c["passes"]]
        failing = [c["payload_length"] for c in ordered if not c["passes"]]
        capacity.append(
            {
                "method": method,
                "max_passing_payload": max(passing) if passing else None,
                "first_failing_payload": min(failing) if failing else None,
                "payloads_tested": [c["payload_length"] for c in ordered],
            }
        )
    capacities = [c["max_passing_payload"] for c in capacity if c["max_passing_payload"] is not None]
    best = max(
        (c for c in capacity if c["max_passing_payload"] is not None),
        key=lambda c: c["max_passing_payload"],
        default=None,
    )
    return {
        "overall": group_stats(rows),
        "thresholds": {"min_bit_accuracy": min_accuracy, "max_ber": max_ber},
        "by_method_payload": cells,
        "capacity_by_method": capacity,
        "best_capacity_method": best["method"] if best else None,
        "highest_stable_payload": max(capacities) if capacities else None,
        "average_capacity": sum(capacities) / len(capacities) if capacities else None,
    }


def _validate(config: ExperimentConfig) -> list[str]:
    if len(config.payload_lengths) < 2:
        return ["Embedding capacity needs at least two payload lengths to sweep."]
    return []


SCENARIO = Scenario(
    experiment_type=ExperimentType.EMBEDDING_CAPACITY,
    title="Embedding Capacity",
    description=(
        "Sweep payload lengths per method and find the largest payload that still meets "
        "the bit-accuracy and BER thresholds."
    ),
    default_payload_lengths=PAYLOAD_PRESETS,
    validate=_validate,
    summarize=_summarize,
)
