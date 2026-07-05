"""Normalized experiment results and aggregate summaries.

Every scenario produces the same flat row shape so that any experiment can be
exported to CSV and compared against any other.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from taf.evaluation.result import EvaluationRow

# Direction heuristics for metric normalization in method comparison: metric
# names matching these fragments are treated as "lower is better".
LOWER_IS_BETTER_FRAGMENTS = ("distance", "cepstrum", "wss", "llr", "bsd", "mel")


class ExperimentResultRow(BaseModel):
    experiment_id: str
    experiment_type: str
    timestamp: datetime
    dataset_id: str | None = None
    file_name: str
    file_path: str
    sample_rate: int | None = None
    duration_seconds: float | None = None
    channels: int = 1
    method: str
    method_type: str | None = None
    payload_length: int
    repetition: int = 0
    message_bits: str | None = None
    decoded_bits: str | None = None
    attack: str | None = None
    attack_parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_errors: dict[str, str] = Field(default_factory=dict)
    bit_accuracy: float | None = None
    ber: float | None = None
    decode_success: bool = False
    encode_time_seconds: float | None = None
    decode_time_seconds: float | None = None
    attack_time_seconds: float | None = None
    total_time_seconds: float | None = None
    status: str = "ok"
    error: str | None = None


def bit_accuracy(original: Sequence[int], decoded: Sequence[int] | None) -> float:
    """Fraction of message bits reproduced at the same position (0..1)."""
    if not original:
        return 0.0
    if decoded is None:
        return 0.0
    correct = sum(
        1
        for index, bit in enumerate(original)
        if index < len(decoded) and int(bit) == int(decoded[index])
    )
    return correct / len(original)


def bit_error_rate(original: Sequence[int], decoded: Sequence[int] | None) -> float:
    """BER = 1 - bit accuracy; missing decoded bits count as errors."""
    return 1.0 - bit_accuracy(original, decoded)


def normalize_row(
    row: EvaluationRow,
    *,
    experiment_id: str,
    experiment_type: str,
    dataset_id: str | None,
    method_descriptions: dict[str, str] | None = None,
) -> ExperimentResultRow:
    """Convert one engine ``EvaluationRow`` into the normalized result shape."""
    bits = row.message_bits or []
    accuracy = bit_accuracy(bits, row.decoded_message) if bits else None
    times = [row.encode_time_seconds, row.decode_time_seconds, row.attack_time_seconds]
    known_times = [value for value in times if value is not None]
    repetition = _repetition_from_message_name(row.message_name)
    return ExperimentResultRow(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        timestamp=datetime.now(timezone.utc),
        dataset_id=dataset_id,
        file_name=Path(row.input_path).name,
        file_path=str(row.input_path),
        sample_rate=row.sample_rate,
        duration_seconds=row.duration_seconds,
        method=row.method,
        method_type=(method_descriptions or {}).get(row.method),
        payload_length=row.message_length,
        repetition=repetition,
        message_bits="".join(str(int(bit)) for bit in bits) if bits else None,
        decoded_bits=(
            "".join(str(int(bit)) for bit in row.decoded_message)
            if row.decoded_message is not None
            else None
        ),
        attack=row.attack,
        metrics=_finite_metrics(row.metrics),
        metric_errors=dict(row.metric_errors),
        bit_accuracy=accuracy,
        ber=1.0 - accuracy if accuracy is not None else None,
        decode_success=row.success,
        encode_time_seconds=row.encode_time_seconds,
        decode_time_seconds=row.decode_time_seconds,
        attack_time_seconds=row.attack_time_seconds,
        total_time_seconds=sum(known_times) if known_times else None,
        status="error" if row.error else "ok",
        error=row.error,
    )


def _repetition_from_message_name(name: str) -> int:
    # Engine message names end with the message index: random_000_len16_002.
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Keep scalar metric values; arrays collapse to their mean; NaN/Inf -> None."""
    normalized: dict[str, float | None] = {}
    for name, value in metrics.items():
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                finite = [float(v) for v in value if v is not None and math.isfinite(float(v))]
                normalized[name] = sum(finite) / len(finite) if finite else None
            else:
                scalar = float(value)
                normalized[name] = scalar if math.isfinite(scalar) else None
        except (TypeError, ValueError):
            normalized[name] = None
    return normalized


# --------------------------------------------------------------------------
# Aggregation helpers (used by scenario summaries)
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _collect(rows: Sequence[ExperimentResultRow], getter) -> list[float]:
    values = []
    for row in rows:
        value = getter(row)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def group_stats(rows: Sequence[ExperimentResultRow]) -> dict[str, Any]:
    """Core statistics for any group of rows."""
    ok_rows = [row for row in rows if row.status == "ok"]
    metric_names = sorted({name for row in rows for name in row.metrics})
    avg_metrics = {
        name: _mean(_collect(rows, lambda r, n=name: r.metrics.get(n))) for name in metric_names
    }
    return {
        "rows": len(rows),
        "error_rows": len(rows) - len(ok_rows),
        "decode_success_rate": _mean([1.0 if row.decode_success else 0.0 for row in rows]),
        "avg_bit_accuracy": _mean(_collect(rows, lambda r: r.bit_accuracy)),
        "avg_ber": _mean(_collect(rows, lambda r: r.ber)),
        "avg_encode_time_seconds": _mean(_collect(rows, lambda r: r.encode_time_seconds)),
        "avg_decode_time_seconds": _mean(_collect(rows, lambda r: r.decode_time_seconds)),
        "avg_metrics": {name: value for name, value in avg_metrics.items() if value is not None},
    }


def group_by(
    rows: Sequence[ExperimentResultRow], key
) -> dict[Any, list[ExperimentResultRow]]:
    grouped: dict[Any, list[ExperimentResultRow]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    return grouped


def by_method(rows: Sequence[ExperimentResultRow]) -> list[dict[str, Any]]:
    return [
        {"method": method, **group_stats(group)}
        for method, group in sorted(group_by(rows, lambda r: r.method).items())
    ]


def by_method_attack(rows: Sequence[ExperimentResultRow]) -> list[dict[str, Any]]:
    grouped = group_by(rows, lambda r: (r.method, r.attack))
    return [
        {"method": method, "attack": attack, **group_stats(group)}
        for (method, attack), group in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1] or "")
        )
    ]


def by_method_payload(rows: Sequence[ExperimentResultRow]) -> list[dict[str, Any]]:
    grouped = group_by(rows, lambda r: (r.method, r.payload_length))
    return [
        {"method": method, "payload_length": payload, **group_stats(group)}
        for (method, payload), group in sorted(grouped.items())
    ]


def metric_is_lower_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return any(fragment in lowered for fragment in LOWER_IS_BETTER_FRAGMENTS)


__all__ = [
    "ExperimentResultRow",
    "bit_accuracy",
    "bit_error_rate",
    "by_method",
    "by_method_attack",
    "by_method_payload",
    "group_by",
    "group_stats",
    "metric_is_lower_better",
    "normalize_row",
]
