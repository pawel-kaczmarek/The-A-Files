"""CSV export for normalized experiment results (pandas-based)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from taf.experiments.results import ExperimentResultRow

# Stable column order for detailed exports; metric columns are appended after.
_BASE_COLUMNS = [
    "experiment_id",
    "experiment_type",
    "timestamp",
    "dataset_id",
    "file_name",
    "file_path",
    "sample_rate",
    "duration_seconds",
    "channels",
    "method",
    "method_type",
    "payload_length",
    "repetition",
    "message_bits",
    "decoded_bits",
    "attack",
    "attack_parameters",
    "bit_accuracy",
    "ber",
    "decode_success",
    "encode_time_seconds",
    "decode_time_seconds",
    "attack_time_seconds",
    "total_time_seconds",
    "status",
    "error",
]


def rows_to_dataframe(rows: Sequence[ExperimentResultRow]) -> pd.DataFrame:
    """Flatten normalized rows: one column per metric, stable base columns."""
    records: list[dict[str, Any]] = []
    metric_names: list[str] = sorted({name for row in rows for name in row.metrics})
    attack_metric_names: list[str] = sorted({name for row in rows for name in row.attack_metrics})
    for row in rows:
        record = row.model_dump(mode="json")
        metrics = record.pop("metrics", {})
        metric_errors = record.pop("metric_errors", {})
        attack_metrics = record.pop("attack_metrics", {})
        attack_metric_errors = record.pop("attack_metric_errors", {})
        for name in attack_metric_names:
            if name in attack_metrics:
                record[f"attack_metric:{name}"] = attack_metrics[name]
            elif name in attack_metric_errors:
                record[f"attack_metric:{name}"] = f"error: {attack_metric_errors[name]}"
            else:
                record[f"attack_metric:{name}"] = None
        # Attack parameters are what make a row reproducible, so they are kept
        # as a JSON string rather than dropped.
        record["attack_parameters"] = (
            json.dumps(record.pop("attack_parameters", None), sort_keys=True, default=str)
            if record.get("attack_parameters")
            else None
        )
        for name in metric_names:
            if name in metrics:
                record[f"metric:{name}"] = metrics[name]
            elif name in metric_errors:
                record[f"metric:{name}"] = f"error: {metric_errors[name]}"
            else:
                record[f"metric:{name}"] = None
        records.append(record)
    columns = (
        _BASE_COLUMNS
        + [f"metric:{name}" for name in metric_names]
        + [f"attack_metric:{name}" for name in attack_metric_names]
    )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.reindex(columns=columns)


def export_detailed_csv(rows: Sequence[ExperimentResultRow]) -> str:
    return rows_to_dataframe(rows).to_csv(index=False, lineterminator="\n")


def summary_to_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten a scenario summary into one long frame.

    Every list of objects becomes its own group of records, labelled by its
    path in the summary (``statistics.ber_baseline.pairwise``); nested scalar
    values become columns named by their path (``ber_stats:mean``). Tables
    nested at any depth, such as the pairwise tests inside a comparison, are
    therefore exported instead of being stringified.
    """
    records: list[dict[str, Any]] = []
    for section, value in summary.items():
        _emit(section, value, records)
    return pd.DataFrame.from_records(records)


def _is_table(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


def _emit(section: str, value: Any, records: list[dict[str, Any]]) -> None:
    if _is_table(value):
        for entry in value:
            flat: dict[str, Any] = {"section": section}
            _flatten_into(flat, "", entry, section, records)
            records.append(flat)
    elif isinstance(value, dict):
        flat = {"section": section}
        _flatten_into(flat, "", value, section, records)
        if len(flat) > 1:
            records.append(flat)
    else:
        records.append({"section": section, "value": _scalar(value)})


def _flatten_into(
    flat: dict[str, Any], prefix: str, value: dict[str, Any], section: str, records: list[dict[str, Any]]
) -> None:
    for key, item in value.items():
        name = f"{prefix}:{key}" if prefix else str(key)
        if _is_table(item):
            _emit(f"{section}.{name.replace(':', '.')}", item, records)
        elif isinstance(item, dict):
            _flatten_into(flat, name, item, section, records)
        else:
            flat[name] = _scalar(item)


def _scalar(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def export_summary_csv(summary: dict[str, Any]) -> str:
    return summary_to_dataframe(summary).to_csv(index=False, lineterminator="\n")


def make_export_filename(
    experiment_type: str,
    experiment_id: str,
    kind: str = "detailed",
    at: datetime | None = None,
) -> str:
    """e.g. dataset_benchmark_ab12cd34ef56_detailed_2026_07_05_143000.csv"""
    stamp = (at or datetime.now(timezone.utc)).strftime("%Y_%m_%d_%H%M%S")
    return f"{experiment_type}_{experiment_id}_{kind}_{stamp}.csv"


__all__ = [
    "export_detailed_csv",
    "export_summary_csv",
    "make_export_filename",
    "rows_to_dataframe",
    "summary_to_dataframe",
]
