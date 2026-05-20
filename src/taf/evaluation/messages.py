from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


__all__ = ["EvaluationMessage", "RandomMessageSpec"]
