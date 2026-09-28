from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taf.audio.formats import DecodeTarget
from taf.evaluation.messages import EvaluationMessage


class FailureKind:
    """Why a trial produced no decoded message.

    A failure is not a bit error: it says nothing about how many bits a
    decoder would have got right. Keeping the reason lets an analysis report
    failures separately from the bit error rate instead of folding them into
    it.
    """

    #: The payload exceeds what the cover can carry (``CapacityError``).
    OVER_CAPACITY = "over_capacity"
    #: ``encode()`` raised for any other reason.
    ENCODE_ERROR = "encode_error"
    #: Writing or re-reading the stego file failed.
    IO_ERROR = "io_error"
    #: The attack itself raised.
    ATTACK_ERROR = "attack_error"
    #: ``decode()`` raised.
    DECODE_ERROR = "decode_error"

    ALL = (OVER_CAPACITY, ENCODE_ERROR, IO_ERROR, ATTACK_ERROR, DECODE_ERROR)


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
    #: One of ``FailureKind``; ``None`` when the trial completed.
    failure_kind: str | None = None
    #: Index of the message within its length, i.e. the repetition.
    repetition: int | None = None
    #: Catalogue name and constructor parameters of the method, when named.
    method_name: str | None = None
    method_parameters: dict[str, Any] = field(default_factory=dict)
    is_lossy: bool = False
    transformation_name: str | None = None
    codec_options: dict[str, Any] = field(default_factory=dict)
    attack: str | None = None
    attack_parameters: dict[str, Any] = field(default_factory=dict)
    #: Quality of the attacked signal measured against the stego signal, i.e.
    #: the damage the attack did. ``metrics`` stays cover-vs-stego.
    attack_metrics: dict[str, Any] = field(default_factory=dict)
    attack_metric_errors: dict[str, str] = field(default_factory=dict)
    message_bits: list[int] | None = None
    sample_rate: int | None = None
    duration_seconds: float | None = None
    encode_time_seconds: float | None = None
    decode_time_seconds: float | None = None
    attack_time_seconds: float | None = None
    channels: int = 1
    sample_count: int | None = None
    audio_metadata: dict[str, Any] = field(default_factory=dict)
    payload_kind: str = "random"
    payload_seed: int | None = None
    payload_metadata: dict[str, Any] = field(default_factory=dict)

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
            "failure_kind": self.failure_kind,
            "repetition": self.repetition,
            "method_name": self.method_name,
            "method_parameters": self.method_parameters,
            "is_lossy": self.is_lossy,
            "transformation_name": self.transformation_name,
            "codec_options": self.codec_options,
            "attack": self.attack,
            "attack_parameters": self.attack_parameters,
            "attack_metrics": self.attack_metrics,
            "attack_metric_errors": self.attack_metric_errors,
            "message_bits": self.message_bits,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "encode_time_seconds": self.encode_time_seconds,
            "decode_time_seconds": self.decode_time_seconds,
            "attack_time_seconds": self.attack_time_seconds,
            "channels": self.channels,
            "sample_count": self.sample_count,
            "audio_metadata": self.audio_metadata,
            "payload_kind": self.payload_kind,
            "payload_seed": self.payload_seed,
            "payload_metadata": self.payload_metadata,
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


__all__ = ["EvaluationResult", "EvaluationRow", "FailureKind"]
