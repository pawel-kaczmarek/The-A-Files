from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taf.audio.formats import DecodeTarget
from taf.evaluation.messages import EvaluationMessage


@dataclass
class EvaluationRow:
    input_path: Path
    method: str
    message_name: str
    message_length: int
    decode_mode: str
    format: str | None
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    metric_errors: dict[str, str] = field(default_factory=dict)
    output_path: Path | None = None
    decoded_message: list[int] | None = None
    error: str | None = None
    is_lossy: bool = False
    transformation_name: str | None = None
    codec_options: dict[str, Any] = field(default_factory=dict)
    attack: str | None = None
    message_bits: list[int] | None = None
    sample_rate: int | None = None
    duration_seconds: float | None = None
    encode_time_seconds: float | None = None
    decode_time_seconds: float | None = None
    attack_time_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "method": self.method,
            "message_name": self.message_name,
            "message_length": self.message_length,
            "decode_mode": self.decode_mode,
            "format": self.format,
            "success": self.success,
            "metrics": self.metrics,
            "metric_errors": self.metric_errors,
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "decoded_message": self.decoded_message,
            "error": self.error,
            "is_lossy": self.is_lossy,
            "transformation_name": self.transformation_name,
            "codec_options": self.codec_options,
            "attack": self.attack,
            "message_bits": self.message_bits,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "encode_time_seconds": self.encode_time_seconds,
            "decode_time_seconds": self.decode_time_seconds,
            "attack_time_seconds": self.attack_time_seconds,
        }


@dataclass
class EvaluationResult:
    messages: dict[str, EvaluationMessage]
    rows: list[EvaluationRow] = field(default_factory=list)

    def success_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(row.success for row in self.rows) / len(self.rows)

    def by_method(self) -> dict[str, list[EvaluationRow]]:
        grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
        for row in self.rows:
            grouped[row.method].append(row)
        return dict(grouped)

    def by_format(self) -> dict[str, list[EvaluationRow]]:
        grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
        for row in self.rows:
            grouped[row.format or DecodeTarget.DIRECT.value].append(row)
        return dict(grouped)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.rows]


__all__ = ["EvaluationRow", "EvaluationResult"]
