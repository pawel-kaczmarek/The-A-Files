"""Gain, clipping and dynamic-range attacks.

These are the cheapest manipulations in the whole taxonomy - a volume slider
and a limiter - which is exactly why they matter. Any scheme whose detector
compares a statistic against a fixed absolute threshold fails after a 3 dB
gain change, while a scheme that decides on a *relation* between two measured
quantities is unaffected. The attack separates those two designs immediately.

Gain is specified in decibels, the unit volume controls actually use:

    gain_linear = 10^(gain_db / 20)

Clipping is kept as a separate attack, because a gain change and a clipped
signal are different phenomena: gain is invertible, clipping destroys
information. Applying gain does not normalise afterwards - normalisation would
undo the attack - but it reports how many samples left full scale so the
experiment can tell gain-induced clipping from the clipping attack proper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from taf.attacks.base import (
    Attack,
    AttackCategory,
    AttackError,
    clip_to_full_scale,
    rms,
)


@dataclass(frozen=True)
class GainChange(Attack):
    """Scale amplitude by a gain in decibels.

    ``prevent_clipping`` is off by default: clipping is part of what a loud
    playback chain does, and silently limiting it would understate the attack.
    Turn it on to measure gain alone.
    """

    gain_db: float = -6.0
    prevent_clipping: bool = False

    name = "gain"
    category = AttackCategory.AMPLITUDE

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        factor = 10.0 ** (self.gain_db / 20.0)
        scaled = audio * factor

        over_full_scale = int(np.count_nonzero(np.abs(scaled) > 1.0))
        clipped = 0
        if self.prevent_clipping:
            scaled, clipped = clip_to_full_scale(scaled)

        input_rms = rms(audio)
        output_rms = rms(scaled)
        measured_gain_db = (
            20.0 * np.log10(output_rms / input_rms) if input_rms > 0 and output_rms > 0 else float("nan")
        )

        return scaled, sample_rate, {
            "gain_linear": factor,
            "measured_gain_db": float(measured_gain_db),
            "samples_over_full_scale": over_full_scale,
            "clipped_samples": clipped,
            "normalized": False,
        }


@dataclass(frozen=True)
class Clipping(Attack):
    """Hard-clip the waveform at a threshold.

    The threshold is interpreted in one of three ways, because the literature
    uses all three and they are not interchangeable:

    * ``"full_scale"`` - an absolute amplitude in [0, 1]. Models a converter
      or amplifier running out of headroom.
    * ``"peak"`` - a fraction of the signal's own peak. Comparable across
      files that were mastered at different levels.
    * ``"percentile"`` - the given percentile of |x|, so the fraction of
      samples actually clipped is fixed by construction. This is the most
      comparable across a heterogeneous dataset.
    """

    threshold: float = 0.9
    mode: str = "peak"

    name = "clipping"
    category = AttackCategory.AMPLITUDE

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        magnitude = np.abs(audio)
        peak = float(np.max(magnitude))
        if peak == 0.0:
            raise AttackError("cannot clip a silent signal")

        if self.mode == "full_scale":
            if not 0.0 < self.threshold <= 1.0:
                raise AttackError("threshold must be in (0, 1] for mode='full_scale'")
            level = float(self.threshold)
        elif self.mode == "peak":
            if not 0.0 < self.threshold <= 1.0:
                raise AttackError("threshold must be in (0, 1] for mode='peak'")
            level = float(self.threshold) * peak
        elif self.mode == "percentile":
            if not 0.0 < self.threshold < 100.0:
                raise AttackError("threshold must be a percentile in (0, 100) for mode='percentile'")
            level = float(np.percentile(magnitude, self.threshold))
        else:
            raise AttackError(
                f"unknown clipping mode {self.mode!r}; expected 'full_scale', 'peak' or 'percentile'"
            )

        if level <= 0.0:
            raise AttackError("computed clipping level is zero; the signal is mostly silent")

        clipped_count = int(np.count_nonzero(magnitude > level))
        clipped = np.clip(audio, -level, level)

        return clipped, sample_rate, {
            "clipping_level": level,
            "clipping_mode": self.mode,
            "clipped_samples": clipped_count,
            "clipped_fraction": clipped_count / audio.size,
            "signal_peak": peak,
        }


@dataclass(frozen=True)
class DynamicRangeCompression(Attack):
    """Static compressor above a threshold, then make-up gain.

    A memoryless (no attack/release) compressor: samples above the threshold
    are scaled down by the ratio, and the whole signal is raised so the peak
    returns to where it was. Broadcast and streaming loudness processing does
    something along these lines, and it is the manipulation most likely to
    disturb amplitude-relation schemes while leaving the audio perfectly
    listenable.

    The memoryless model is a deliberate simplification: it is fully specified
    by two numbers and therefore reproducible, where a real compressor's
    envelope follower would add unstated time constants.
    """

    threshold: float = 0.3
    ratio: float = 4.0

    name = "compression_dynamic"
    category = AttackCategory.AMPLITUDE

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if not 0.0 < self.threshold < 1.0:
            raise AttackError(f"threshold must be in (0, 1), got {self.threshold}")
        if self.ratio < 1.0:
            raise AttackError(f"ratio must be at least 1, got {self.ratio}")

        magnitude = np.abs(audio)
        peak_before = float(np.max(magnitude))
        if peak_before == 0.0:
            raise AttackError("cannot compress a silent signal")

        above = magnitude > self.threshold
        compressed_magnitude = np.where(
            above,
            self.threshold + (magnitude - self.threshold) / self.ratio,
            magnitude,
        )
        compressed = np.sign(audio) * compressed_magnitude

        peak_after = float(np.max(np.abs(compressed)))
        make_up = peak_before / peak_after if peak_after > 0 else 1.0
        compressed = compressed * make_up

        return compressed, sample_rate, {
            "threshold": self.threshold,
            "ratio": self.ratio,
            "samples_compressed": int(np.count_nonzero(above)),
            "make_up_gain_linear": float(make_up),
            "make_up_gain_db": float(20.0 * np.log10(make_up)) if make_up > 0 else float("nan"),
        }


__all__ = ["GainChange", "Clipping", "DynamicRangeCompression"]
