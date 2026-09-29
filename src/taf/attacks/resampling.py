"""Sample-rate conversion attacks.

Sample-rate conversion is ubiquitous: every format change, every playback
device with a different clock, every streaming pipeline does it. It combines
two effects that damage watermarks in different ways - a band limit at the
lower Nyquist frequency, and an interpolation of the waveform onto a new grid,
which perturbs individual sample values everywhere even in the retained band.

The attack is a **round trip**: the signal goes down (or up) to an intermediate
rate and comes back to the original one, so the decoder receives audio at the
rate it expects. This is the important correction over the previous
implementation, which resampled once and left the signal at the new rate: what
that measured was whether the decoder could cope with an unexpected sample
rate - an API mismatch - rather than whether the payload survived conversion.

Conversion uses ``scipy.signal.resample_poly``, a polyphase FIR rational
resampler with an anti-aliasing filter built in. Naive decimation without that
filter folds high-frequency content back into the band and would make the
attack look far more destructive than real resampling is.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from taf.attacks.base import Attack, AttackCategory, AttackError, per_channel
from taf.models.card import AttackCard, Text


def resample_to(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resample with anti-aliasing, preserving the channel layout."""
    if source_rate == target_rate:
        return audio

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return per_channel(audio, lambda channel: resample_poly(channel, up, down))


@dataclass(frozen=True)
class ResampleRoundTrip(Attack):
    """Convert to an intermediate sample rate and back.

    Args:
        intermediate_hz: The rate the signal passes through. Values below the
            original rate impose a band limit at the intermediate Nyquist
            frequency and are the destructive case; values above it only
            interpolate.
        restore_length: Trim or pad the returned signal to the original length.
            Rational resampling can land a sample or two off, which would
            desynchronise a block-based decoder for reasons that have nothing
            to do with the conversion itself. The correction is recorded.
    """

    intermediate_hz: int = 22050
    restore_length: bool = True

    name = "resample"
    category = AttackCategory.RESAMPLING
    card = AttackCard(
        title=Text("Resampling round trip", "Zmiana częstotliwości próbkowania"),
        summary=Text(
            en="Converts to an intermediate sample rate and back to test rate-conversion damage.",
            pl="Zmienia częstotliwość próbkowania na pośrednią i z powrotem, badając skutki konwersji.",
        ),
        details=Text(
            en=(
                "Uses polyphase FIR resampling with anti-alias filtering. A lower intermediate_hz "
                "limits the recoverable bandwidth and changes the sample grid. Optional restore_length "
                "trims or pads rounding differences. Returning to the original rate lets the decoder "
                "use its expected rate, but does not restore discarded frequencies or original sample "
                "values."
            ),
            pl=(
                "Używa wielofazowego resamplingu FIR z filtracją antyaliasingową. Niższy "
                "intermediate_hz ogranicza zachowane pasmo i zmienia siatkę próbek. Opcjonalny "
                "restore_length przycina lub uzupełnia różnice zaokrągleń. Powrót do pierwotnej "
                "częstotliwości daje dekoderowi oczekiwany format, ale nie odtwarza odrzuconych "
                "częstotliwości ani wartości próbek."
            ),
        ),
    )
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if self.intermediate_hz <= 0:
            raise AttackError(f"intermediate_hz must be positive, got {self.intermediate_hz}")

        original_length = audio.shape[0]
        downsampled = resample_to(audio, sample_rate, int(self.intermediate_hz))
        restored = resample_to(downsampled, int(self.intermediate_hz), sample_rate)

        length_correction = 0
        if self.restore_length and restored.shape[0] != original_length:
            length_correction = original_length - restored.shape[0]
            if length_correction > 0:
                pad_width = [(0, length_correction)] + [(0, 0)] * (restored.ndim - 1)
                restored = np.pad(restored, pad_width)
            else:
                restored = restored[:original_length]

        return restored, sample_rate, {
            "intermediate_sample_rate": int(self.intermediate_hz),
            "intermediate_length": int(downsampled.shape[0]),
            "algorithm": "scipy.signal.resample_poly",
            "anti_aliased": True,
            "band_limit_hz": float(min(self.intermediate_hz, sample_rate) / 2.0),
            "length_correction_samples": int(length_correction),
            "restore_length": bool(self.restore_length),
        }


@dataclass(frozen=True)
class SampleRateOffset(Attack):
    """A small clock mismatch between playback and capture.

    Two devices never share a clock exactly. A drift of a few hundred parts
    per million is inaudible, leaves the spectrum essentially untouched, and
    accumulates into a sample-level misalignment over the length of a clip -
    which is precisely what a block-synchronised decoder cannot tolerate. It
    is implemented as resampling by (1 + offset_ppm/1e6) with the nominal rate
    left unchanged, because that is what the receiving device believes.
    """

    offset_ppm: float = 100.0

    name = "clock_drift"
    category = AttackCategory.RESAMPLING
    card = AttackCard(
        title=Text("Clock drift", "Dryft zegara"),
        summary=Text(
            en="Simulates a small mismatch between playback and recording clocks.",
            pl="Symuluje małą różnicę zegarów urządzenia odtwarzającego i nagrywającego.",
        ),
        details=Text(
            en=(
                "Resamples by a factor derived from offset_ppm while leaving the nominal sample rate "
                "unchanged. Even a small mismatch accumulates into positional drift over a long "
                "recording. The attack tests whether a decoder can track a gradually changing sample "
                "grid, not simply a constant initial offset."
            ),
            pl=(
                "Próbkuje ponownie ze współczynnikiem wynikającym z offset_ppm, pozostawiając nominalną "
                "częstotliwość bez zmian. Nawet mała różnica narasta do zauważalnego przesunięcia "
                "próbek w długim nagraniu. Atak sprawdza śledzenie stopniowo zmieniającej się siatki, a "
                "nie tylko stałego przesunięcia początku."
            ),
        ),
    )
    changes_length_or_rate = True

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        if self.offset_ppm == 0.0:
            return audio, sample_rate, {"offset_ppm": 0.0, "drift_samples": 0}

        factor = 1.0 + self.offset_ppm / 1e6
        # Express the drift as an exact integer ratio so the conversion is
        # reproducible rather than dependent on float rounding.
        up = int(round(factor * 1_000_000))
        down = 1_000_000
        divisor = gcd(up, down)

        drifted = per_channel(audio, lambda channel: resample_poly(channel, up // divisor, down // divisor))

        return drifted, sample_rate, {
            "offset_ppm": float(self.offset_ppm),
            "resample_ratio": f"{up // divisor}/{down // divisor}",
            "drift_samples": int(drifted.shape[0] - audio.shape[0]),
            "algorithm": "scipy.signal.resample_poly",
        }


__all__ = ["ResampleRoundTrip", "SampleRateOffset", "resample_to"]
