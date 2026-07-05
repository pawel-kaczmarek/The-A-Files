"""CSV export for normalized experiment results (pandas-based)."""

from __future__ import annotations

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
    for row in rows:
        record = row.model_dump(mode="json")
        metrics = record.pop("metrics", {})
        metric_errors = record.pop("metric_errors", {})
        record.pop("attack_parameters", None)
        for name in metric_names:
            if name in metrics:
                record[f"metric:{name}"] = metrics[name]
            elif name in metric_errors:
                record[f"metric:{name}"] = f"error: {metric_errors[name]}"
            else:
                record[f"metric:{name}"] = None
        records.append(record)
    columns = _BASE_COLUMNS + [f"metric:{name}" for name in metric_names]
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.reindex(columns=columns)


def export_detailed_csv(rows: Sequence[ExperimentResultRow]) -> str:
    return rows_to_dataframe(rows).to_csv(index=False, lineterminator="\n")


def summary_to_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten the per-group tables of a scenario summary into one frame."""
    records: list[dict[str, Any]] = []
    for section, value in summary.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            for entry in value:
                flat = {"section": section}
                for key, item in entry.items():
                    if isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            flat[f"{key}:{sub_key}"] = sub_value
                    elif isinstance(item, list):
                        flat[key] = ";".join(str(v) for v in item)
                    else:
                        flat[key] = item
                records.append(flat)
        elif isinstance(value, dict):
            flat = {"section": section}
            for key, item in value.items():
                if isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        flat[f"{key}:{sub_key}"] = sub_value
                else:
                    flat[key] = item
            records.append(flat)
        else:
            records.append({"section": section, "value": value})
    return pd.DataFrame.from_records(records)


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
