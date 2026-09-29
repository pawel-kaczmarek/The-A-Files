"""Bit-depth reduction (PCM requantisation).

Requantisation is the attack that sample-domain embedding cannot survive: an
LSB payload lives in exactly the bits a 16-to-8-bit conversion throws away.
It is also completely routine - it happens whenever audio is stored at a lower
depth or passed through a converter - which makes it the most realistic threat
to that family of methods.

The quantiser is explicit rather than implied by rounding. For a depth of *b*
bits over the nominal range [-1, 1] the step is

    delta = 2 / (2^b)

and the two grid conventions differ in whether zero is a level:

* mid-tread:  q(x) = round(x / delta) * delta     (zero is a level)
* mid-riser:  q(x) = (floor(x / delta) + 0.5) * delta

Mid-tread is the default because it is what integer PCM conversion does.
Optional TPDF dither is available, since dithered and undithered conversion
are different channels: dither removes the correlation between the
quantisation error and the signal, and it also destroys an LSB payload more
thoroughly than plain truncation does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from taf.attacks.base import Attack, AttackCategory, AttackError
from taf.models.card import AttackCard, Text


@dataclass(frozen=True)
class BitDepthReduction(Attack):
    """Uniform requantisation to a lower PCM bit depth.

    Args:
        bits: Target depth. 16, 12 and 8 are the interesting steps: 16 is
            transparent for most embedding, 12 is the borderline case, and 8
            removes half the mantissa of a 16-bit sample.
        dither: Add TPDF dither of one LSB before quantising.
        mode: ``"mid_tread"`` (zero is a quantisation level, as in integer
            PCM) or ``"mid_riser"``.
        seed: Seed for the dither generator.
    """

    bits: int = 8
    dither: bool = False
    mode: str = "mid_tread"
    seed: int | None = 0

    name = "bit_depth"
    category = AttackCategory.QUANTIZATION
    card = AttackCard(
        title=Text("Bit-depth reduction", "Redukcja głębi bitowej"),
        summary=Text(
            en="Requantises audio to fewer PCM amplitude levels.",
            pl="Ponownie kwantuje dźwięk do mniejszej liczby poziomów amplitudy PCM.",
        ),
        details=Text(
            en=(
                "Maps samples to a uniform grid with spacing 2 / 2^bits over nominal full scale. The "
                "mode selects a grid with or without an exact zero level; optional TPDF dither adds "
                "noise before quantisation. Lower bit depth removes finer sample detail and is "
                "particularly destructive to payloads stored in low sample bits."
            ),
            pl=(
                "Przypisuje próbki do równomiernej siatki o odstępie 2 / 2^bits w nominalnym zakresie. "
                "Tryb wybiera siatkę z dokładnym poziomem zera lub bez niego; opcjonalny dither TPDF "
                "dodaje szum przed kwantyzacją. Mniejsza głębia usuwa drobniejsze szczegóły i "
                "szczególnie niszczy dane zapisane w najmłodszych bitach próbek."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if not 2 <= self.bits <= 24:
            raise AttackError(f"bits must be in [2, 24], got {self.bits}")

        levels = 2 ** int(self.bits)
        step = 2.0 / levels

        working = np.clip(audio, -1.0, 1.0)

        if self.dither:
            rng = np.random.default_rng(self.seed)
            # TPDF dither: the sum of two independent uniform LSB-wide terms.
            noise = rng.uniform(-0.5, 0.5, working.shape) + rng.uniform(-0.5, 0.5, working.shape)
            working = working + noise * step

        if self.mode == "mid_tread":
            quantized = np.rint(working / step) * step
        elif self.mode == "mid_riser":
            quantized = (np.floor(working / step) + 0.5) * step
        else:
            raise AttackError(
                f"unknown quantiser mode {self.mode!r}; expected 'mid_tread' or 'mid_riser'"
            )

        quantized = np.clip(quantized, -1.0, 1.0)
        error = quantized - audio

        return quantized, sample_rate, {
            "bits": int(self.bits),
            "levels": int(levels),
            "step": float(step),
            "quantizer_mode": self.mode,
            "dither": bool(self.dither),
            "distinct_levels_used": int(np.unique(np.rint(quantized / step)).size),
            "max_absolute_error": float(np.max(np.abs(error))),
        }


__all__ = ["BitDepthReduction"]
