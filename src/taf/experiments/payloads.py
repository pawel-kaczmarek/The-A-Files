"""Exact payload representations; no implicit padding, truncation, framing or ECC."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, model_validator

from taf.evaluation.messages import EvaluationMessage, bits_for_rate


class PayloadSpec(BaseModel):
    kind: Literal["random", "text", "binary", "bits"] = "random"
    # Binary is hexadecimal, preserving NULs and leading zero bytes in JSON.
    value: str | None = None

    @model_validator(mode="after")
    def validate_content(self):
        if self.kind == "random":
            if self.value is not None:
                raise ValueError("Random payloads cannot have explicit content.")
        else:
            bits = self.bits()
            if not 1 <= len(bits) <= 8192:
                raise ValueError("Explicit payloads must contain 1–8192 bits.")
        return self

    def bits(self) -> tuple[int, ...]:
        value = self.value or ""
        if self.kind == "bits":
            if not value or set(value) - {"0", "1"}:
                raise ValueError("Bit payloads contain only 0 and 1 (no whitespace).")
            return tuple(int(bit) for bit in value)
        if self.kind == "text":
            data = value.encode("utf-8")
        elif self.kind == "binary":
            try:
                data = bytes.fromhex(value)
            except ValueError as error:
                raise ValueError("Binary payloads must be hexadecimal bytes.") from error
        else:
            raise ValueError("Random payloads are materialized with the experiment seed.")
        return tuple((byte >> shift) & 1 for byte in data for shift in range(7, -1, -1))

    def messages(self, repetitions: int) -> list[EvaluationMessage]:
        bits = self.bits()
        return [EvaluationMessage(
            name=f"{self.kind}_len{len(bits)}_{index:03d}", bits=bits,
            source=self.kind, index=index,
            metadata={"encoding": "utf-8" if self.kind == "text" else "msb-first"},
        ) for index in range(repetitions)]


def payload_digest(bits) -> str:
    """SHA-256 of ASCII '0'/'1'; unambiguous for non-byte-aligned payloads."""
    return hashlib.sha256("".join(str(int(bit)) for bit in bits).encode("ascii")).hexdigest()
