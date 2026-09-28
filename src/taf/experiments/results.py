"""Normalized experiment results and aggregate summaries.

Every scenario produces the same flat row shape so that any experiment can be
exported to CSV and compared against any other.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from taf.evaluation.result import EvaluationRow
from taf.experiments.analysis import FAILURE_IMPUTED_BER, estimate, file_of, trial_ber


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
    source_channels: int | None = None
    bit_depth: int | None = None
    audio_subtype: str | None = None
    audio_category: str | None = None
    audio_source: str | None = None
    audio_sha256: str | None = None
    file_id: str | None = None
    sample_count: int | None = None
    preprocessing: list[str] = Field(default_factory=list)
    method: str
    method_type: str | None = None
    #: Constructor parameters the method ran with (empty: its defaults).
    method_parameters: dict[str, Any] = Field(default_factory=dict)
    payload_length: int
    #: Payload per second of cover audio, so capacities of files of different
    #: lengths can be compared.
    payload_rate_bps: float | None = None
    payload_kind: str | None = None
    payload_seed: int | None = None
    payload_sha256: str | None = None
    payload_bytes: float | None = None
    requested_payload_rate_bps: float | None = None
    payload_bits_per_sample: float | None = None
    #: Exact-message delivered bits per cover second, zero on any failed delivery.
    exact_goodput_bps: float | None = None
    decoded_length: int | None = None
    encode_rtf: float | None = None
    decode_rtf: float | None = None
    repetition: int = 0
    message_bits: str | None = None
    decoded_bits: str | None = None
    attack: str | None = None
    attack_parameters: dict[str, Any] = Field(default_factory=dict)
    #: Cover vs stego: how audible the embedding is. Independent of any attack.
    metrics: dict[str, float | None] = Field(default_factory=dict)
    metric_errors: dict[str, str] = Field(default_factory=dict)
    #: Stego vs attacked: how much the attack degraded the signal.
    attack_metrics: dict[str, float | None] = Field(default_factory=dict)
    attack_metric_errors: dict[str, str] = Field(default_factory=dict)
    #: ``None`` when the trial failed: a failure is not a bit error.
    bit_accuracy: float | None = None
    ber: float | None = None
    decode_success: bool = False
    encode_time_seconds: float | None = None
    decode_time_seconds: float | None = None
    attack_time_seconds: float | None = None
    total_time_seconds: float | None = None
    status: str = "ok"
    #: Why the trial failed (``taf.evaluation.result.FailureKind``).
    failure_kind: str | None = None
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
    failed = row.error is not None
    # A failed trial has no decoded message, so it has no bit error rate
    # either. Scoring it as BER 1.0 - every bit wrong - counted a crash as
    # worse than guessing and dragged the averages of the affected methods.
    accuracy = bit_accuracy(bits, row.decoded_message) if bits and not failed else None
    times = [row.encode_time_seconds, row.decode_time_seconds, row.attack_time_seconds]
    known_times = [value for value in times if value is not None]
    repetition = (
        row.repetition if row.repetition is not None else _repetition_from_message_name(row.message_name)
    )
    from taf.experiments.payloads import payload_digest

    audio = row.audio_metadata
    duration = row.duration_seconds
    samples = row.sample_count or (round(duration * row.sample_rate) if duration and row.sample_rate else None)
    return ExperimentResultRow(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        timestamp=datetime.now(timezone.utc),
        dataset_id=dataset_id,
        file_name=Path(row.input_path).name,
        file_path=str(row.input_path),
        sample_rate=row.sample_rate,
        duration_seconds=row.duration_seconds,
        channels=row.channels,
        source_channels=audio.get("channels"),
        bit_depth=audio.get("bit_depth"),
        audio_subtype=audio.get("subtype"),
        audio_category=audio.get("category"),
        audio_source=audio.get("source"),
        audio_sha256=audio.get("sha256"),
        file_id=audio.get("file_id"),
        sample_count=samples,
        preprocessing=audio.get("preprocessing", []),
        method=row.method,
        method_type=row.method_name or (method_descriptions or {}).get(row.method),
        method_parameters=dict(row.method_parameters or {}),
        payload_length=row.message_length,
        payload_kind=row.payload_kind,
        payload_seed=row.payload_seed,
        payload_sha256=payload_digest(bits) if bits else None,
        payload_bytes=row.message_length / 8,
        requested_payload_rate_bps=row.payload_metadata.get("requested_rate_bps"),
        payload_bits_per_sample=row.message_length / (samples * row.channels) if samples else None,
        exact_goodput_bps=(row.message_length / duration if row.success and not failed else 0.0) if duration else None,
        decoded_length=len(row.decoded_message) if row.decoded_message is not None else None,
        encode_rtf=row.encode_time_seconds / duration if row.encode_time_seconds is not None and duration else None,
        decode_rtf=row.decode_time_seconds / duration if row.decode_time_seconds is not None and duration else None,
        payload_rate_bps=(
            row.message_length / row.duration_seconds if row.duration_seconds else None
        ),
        repetition=repetition,
        message_bits="".join(str(int(bit)) for bit in bits) if bits else None,
        decoded_bits=(
            "".join(str(int(bit)) for bit in row.decoded_message)
            if row.decoded_message is not None
            else None
        ),
        attack=row.attack,
        attack_parameters=dict(row.attack_parameters or {}),
        metrics=_finite_metrics(row.metrics),
        attack_metrics=_finite_metrics(row.attack_metrics),
        attack_metric_errors=dict(row.attack_metric_errors or {}),
        metric_errors=dict(row.metric_errors),
        bit_accuracy=accuracy,
        ber=1.0 - accuracy if accuracy is not None else None,
        decode_success=row.success,
        encode_time_seconds=row.encode_time_seconds,
        decode_time_seconds=row.decode_time_seconds,
        attack_time_seconds=row.attack_time_seconds,
        total_time_seconds=sum(known_times) if known_times else None,
        status="error" if failed else "ok",
        failure_kind=(row.failure_kind or "unknown") if failed else None,
        error=row.error,
    )


def _repetition_from_message_name(name: str) -> int:
    # Engine message names end with the message index: random_000_len16_002.
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Scalar metric values; NaN/Inf -> None.

    The engine already splits multi-valued metrics into named components. An
    array that still arrives here is split into numbered entries rather than
    averaged, since its entries need not measure the same thing.
    """
    normalized: dict[str, float | None] = {}
    for name, value in metrics.items():
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    normalized[name] = _finite_or_none(value[0])
                else:
                    for index, entry in enumerate(value):
                        normalized[f"{name} [{index}]"] = _finite_or_none(entry)
            else:
                normalized[name] = _finite_or_none(value)
        except (TypeError, ValueError):
            normalized[name] = None
    return normalized


def _finite_or_none(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


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


#: BER at or below which a payload is usually still recoverable with an
#: error-correcting code. It is an engineering threshold for reporting, not a
#: property of any method, and is stated explicitly so results can be compared.
USABLE_BER_THRESHOLD = 0.1


def distribution_stats(
    values: Sequence[float], clusters: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Mean, median, spread and range, with a 95% interval for the mean.

    A single mean hides everything that matters in a robustness experiment:
    a method that fails on one file in ten and one that degrades slightly
    everywhere can share an average BER. The median and the range separate
    them.

    The interval is a cluster bootstrap over ``clusters`` (normally the file
    of each value); without clusters every value is its own cluster. The
    normal approximation used before assumed independent, unbounded values,
    while BER is bounded, skewed towards zero and correlated within a file.
    """
    pairs = [
        (float(value), clusters[index] if clusters is not None else index)
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    if not pairs:
        return {
            "count": 0,
            "clusters": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    ordered = sorted(value for value, _ in pairs)
    count = len(ordered)
    mean = sum(ordered) / count
    middle = count // 2
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    variance = sum((value - mean) ** 2 for value in ordered) / (count - 1) if count > 1 else 0.0
    interval = estimate([value for value, _ in pairs], [cluster for _, cluster in pairs])
    # Quartiles by linear interpolation (Hyndman & Fan type 7, numpy's default).
    q1, q3 = (float(value) for value in np.quantile(ordered, [0.25, 0.75]))

    return {
        "count": count,
        "clusters": interval["clusters"],
        "mean": mean,
        "median": median,
        "std": math.sqrt(variance),
        "min": ordered[0],
        "max": ordered[-1],
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "ci95_low": interval["ci95_low"],
        "ci95_high": interval["ci95_high"],
    }


def group_stats(rows: Sequence[ExperimentResultRow]) -> dict[str, Any]:
    """Core statistics for any group of rows.

    Bit-level figures (``avg_ber``, ``ber_stats``, the extraction rates) are
    over completed trials only; how many trials completed is reported next to
    them. ``ber_imputed`` folds failures in at chance level and is the figure
    rankings use, so a method cannot improve its BER by crashing on the hard
    cases.
    """
    ok_rows = [row for row in rows if row.status == "ok"]
    metric_names = sorted({name for row in rows for name in row.metrics})
    avg_metrics = {
        name: _mean(_collect(rows, lambda r, n=name: r.metrics.get(n))) for name in metric_names
    }
    ber_values = [row.ber for row in ok_rows if row.ber is not None]
    ber_clusters = [file_of(row) for row in ok_rows if row.ber is not None]
    failures: dict[str, int] = {}
    for row in rows:
        if row.status != "ok":
            failures[row.failure_kind or "unknown"] = failures.get(row.failure_kind or "unknown", 0) + 1

    clusters = [file_of(row) for row in rows]
    imputed = estimate([trial_ber(row, impute_failures=True) for row in rows], clusters)

    return {
        "rows": len(rows),
        "error_rows": len(rows) - len(ok_rows),
        "completion_rate": len(ok_rows) / len(rows) if rows else None,
        "failures": failures,
        "decode_success_rate": _mean([1.0 if row.decode_success else 0.0 for row in rows]),
        "decode_success": estimate([1.0 if row.decode_success else 0.0 for row in rows], clusters),
        "avg_bit_accuracy": _mean([1.0 - value for value in ber_values]),
        "avg_ber": _mean(ber_values),
        "ber_stats": distribution_stats(ber_values, ber_clusters),
        "ber_imputed": imputed,
        "avg_ber_imputed": imputed["estimate"],
        "avg_bit_accuracy_imputed": (
            1.0 - imputed["estimate"] if imputed["estimate"] is not None else None
        ),
        "failure_imputed_ber": FAILURE_IMPUTED_BER,
        # Share of completed runs that recovered the payload exactly, and the
        # share that stayed within the usable threshold. Both say more about a
        # method than the mean does.
        "perfect_extraction_rate": (
            sum(1 for value in ber_values if value == 0.0) / len(ber_values)
            if ber_values
            else None
        ),
        "usable_extraction_rate": (
            sum(1 for value in ber_values if value <= USABLE_BER_THRESHOLD) / len(ber_values)
            if ber_values
            else None
        ),
        "usable_ber_threshold": USABLE_BER_THRESHOLD,
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


def baseline_rows(rows: Sequence[ExperimentResultRow]) -> list[ExperimentResultRow]:
    """Rows decoded without an attack."""
    return [row for row in rows if row.attack is None]


def attacked_rows(rows: Sequence[ExperimentResultRow]) -> list[ExperimentResultRow]:
    return [row for row in rows if row.attack is not None]


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


def metric_direction(metric_name: str) -> bool | None:
    """``True`` if higher is better, ``False`` if lower is, ``None`` if unknown.

    Taken from the metric's own declaration (``Metric.higher_is_better`` and
    its components), not guessed from its name.
    """
    return _metric_directions().get(metric_name)


def metric_is_lower_better(metric_name: str) -> bool:
    return metric_direction(metric_name) is False


_DIRECTIONS: dict[str, bool | None] | None = None


def _metric_directions() -> dict[str, bool | None]:
    global _DIRECTIONS
    if _DIRECTIONS is None:
        from taf.plugins import metric_directions

        _DIRECTIONS = metric_directions()
    return _DIRECTIONS


__all__ = [
    "ExperimentResultRow",
    "attacked_rows",
    "baseline_rows",
    "bit_accuracy",
    "bit_error_rate",
    "by_method",
    "by_method_attack",
    "by_method_payload",
    "distribution_stats",
    "USABLE_BER_THRESHOLD",
    "group_by",
    "group_stats",
    "metric_direction",
    "metric_is_lower_better",
    "normalize_row",
]
