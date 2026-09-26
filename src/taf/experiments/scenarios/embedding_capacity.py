"""Embedding capacity: practical maximum payload per method.

Capacity is measured per file, because it depends on the cover: a method
that stores one bit per frame carries twice as much in a clip twice as long.
For every (method, file) the capacity is the largest tested payload below the
first one that fails, where a payload passes when every trial completed and
the mean bit accuracy and BER meet the thresholds. Dividing by the duration
of the file gives a rate in bits per second, which is comparable across
files; the per-file values are then summarised with a bootstrap interval.

Payloads above the tested range are not explored: a method that passes at the
largest tested payload has a capacity of *at least* that value, which the
summary marks as ``censored``.
"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.analysis import estimate
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
PAYLOAD_PRESETS = (4, 8, 16, 32, 64, 120, 256, 512, 1024)


def thresholds_from(config: ExperimentConfig) -> tuple[float, float]:
    options = config.advanced_options or {}
    return (
        float(options.get("min_bit_accuracy", DEFAULT_MIN_BIT_ACCURACY)),
        float(options.get("max_ber", DEFAULT_MAX_BER)),
    )


def _passes(stats: dict[str, Any], min_accuracy: float, max_ber: float) -> bool:
    accuracy, ber = stats["avg_bit_accuracy"], stats["avg_ber"]
    return (
        stats["completion_rate"] == 1.0
        and accuracy is not None
        and ber is not None
        and accuracy >= min_accuracy
        and ber <= max_ber
    )


def _capacity(passing_by_payload: dict[int, bool]) -> tuple[int | None, int | None]:
    """(largest payload below the first failure, first failing payload)."""
    capacity: int | None = None
    for payload in sorted(passing_by_payload):
        if not passing_by_payload[payload]:
            return capacity, payload
        capacity = payload
    return capacity, None


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    min_accuracy, max_ber = thresholds_from(config)
    # Capacity is a property of the clean channel; attacked rows belong to
    # a robustness analysis.
    clean = [row for row in rows if row.attack is None]

    cells = by_method_payload(clean)
    for cell in cells:
        cell["passes"] = _passes(cell, min_accuracy, max_ber)

    capacity: list[dict[str, Any]] = []
    for method, method_rows in sorted(group_by(clean, lambda r: r.method).items()):
        pooled = {
            cell["payload_length"]: cell["passes"] for cell in cells if cell["method"] == method
        }
        max_passing, first_failing = _capacity(pooled)

        per_file_bits: list[float] = []
        per_file_bps: list[float] = []
        files: list[str] = []
        censored = 0
        for file_name, file_rows in sorted(group_by(method_rows, lambda r: r.file_name).items()):
            passing = {
                payload: _passes(group_stats(group), min_accuracy, max_ber)
                for payload, group in group_by(file_rows, lambda r: r.payload_length).items()
            }
            bits, failing = _capacity(passing)
            if failing is None:
                censored += 1
            duration = file_rows[0].duration_seconds
            per_file_bits.append(float(bits or 0))
            if duration:
                per_file_bps.append((bits or 0) / duration)
                files.append(file_name)

        over_capacity = sum(1 for row in method_rows if row.failure_kind == "over_capacity")
        bps = estimate(per_file_bps, files)
        capacity.append(
            {
                "method": method,
                # Pooled over files: the payload every file carries.
                "max_passing_payload": max_passing,
                "first_failing_payload": first_failing,
                "payloads_tested": sorted(pooled),
                "files": len(per_file_bits),
                "capacity_bits_median": _median(per_file_bits),
                "capacity_bps_median": _median(per_file_bps),
                "capacity_bps_mean": bps["estimate"],
                "capacity_bps_ci95_low": bps["ci95_low"],
                "capacity_bps_ci95_high": bps["ci95_high"],
                # Files on which even the largest tested payload passed.
                "censored_files": censored,
                "over_capacity_rows": over_capacity,
            }
        )
    capacities = [c["max_passing_payload"] for c in capacity if c["max_passing_payload"] is not None]
    best = max(
        (c for c in capacity if c["capacity_bps_median"] is not None),
        key=lambda c: c["capacity_bps_median"],
        default=None,
    )
    return {
        "overall": group_stats(clean),
        "thresholds": {"min_bit_accuracy": min_accuracy, "max_ber": max_ber},
        "by_method_payload": cells,
        "capacity_by_method": capacity,
        "best_capacity_method": best["method"] if best else None,
        "highest_stable_payload": max(capacities) if capacities else None,
        "average_capacity": sum(capacities) / len(capacities) if capacities else None,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _validate(config: ExperimentConfig) -> list[str]:
    if len(config.payload_lengths) < 2:
        return ["Embedding capacity needs at least two payload lengths to sweep."]
    return []


SCENARIO = Scenario(
    experiment_type=ExperimentType.EMBEDDING_CAPACITY,
    title="Embedding Capacity",
    description=(
        "Sweep payload lengths per method and find, per file, the largest payload that "
        "still meets the bit-accuracy and BER thresholds, in bits and bits per second."
    ),
    property="capacity",
    factors=("method", "payload_length"),
    measures=("ber", "completion_rate", "capacity_bits", "capacity_bps"),
    analyses=("per_file_capacity", "bootstrap_ci"),
    default_payload_lengths=PAYLOAD_PRESETS,
    validate=_validate,
    summarize=_summarize,
)
