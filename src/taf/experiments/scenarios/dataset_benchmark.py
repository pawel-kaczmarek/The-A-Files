"""Dataset benchmark: the broad, general-purpose scenario."""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    by_method,
    by_method_attack,
    by_method_payload,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.schema import ExperimentConfig, ExperimentType


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    methods = by_method(rows)
    ranking = sorted(
        methods,
        key=lambda entry: (entry["avg_bit_accuracy"] is None, -(entry["avg_bit_accuracy"] or 0.0)),
    )
    summary: dict[str, Any] = {
        "overall": group_stats(rows),
        "by_method": methods,
        "method_ranking": [entry["method"] for entry in ranking],
        "by_method_payload": by_method_payload(rows),
    }
    if config.attacks:
        summary["by_method_attack"] = by_method_attack(rows)
    return summary


SCENARIO = Scenario(
    experiment_type=ExperimentType.DATASET_BENCHMARK,
    title="Dataset Benchmark",
    description=(
        "Run selected steganography methods over a dataset for chosen payload lengths, "
        "metrics and optional attacks — the general benchmark scenario."
    ),
    summarize=_summarize,
)
