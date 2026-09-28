from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass(frozen=True)
class EvaluationMessage:
    name: str
    bits: tuple[int, ...]
    source: str = "manual"
    seed: int | None = None
    index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.bits)


@dataclass(frozen=True)
class RandomMessageSpec:
    length: int
    count: int = 1
    seed: int | None = None
    name_prefix: str = "random"


def bits_for_rate(rate: float, frames: int, sample_rate: int) -> int:
    """Floor requested bps × duration, without silent clamping or padding."""
    if not math.isfinite(rate) or rate <= 0 or sample_rate <= 0:
        raise ValueError("Payload rate and sample rate must be finite and positive.")
    length = math.floor(rate * frames / sample_rate)
    if not 1 <= length <= 8192:
        raise ValueError(f"Rate {rate:g} bps resolves to {length} bits; supported range is 1–8192.")
    return length


__all__ = ["EvaluationMessage", "RandomMessageSpec", "bits_for_rate"]
