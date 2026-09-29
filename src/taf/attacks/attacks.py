"""Chainable attack builder.

``CorruptedWavFile`` is the fluent front end to the attack package. Every
method here delegates to an implementation in :mod:`taf.attacks` and records
the resulting metadata on :attr:`CorruptedWavFile.applied`, so a chain can be
reported in full afterwards.

Several methods changed behaviour when the attacks were put on a proper
signal-processing footing, and the differences are documented on each one.
The notable ones:

* ``additive_noise`` is now specified by target SNR, not by an absolute
  standard deviation whose severity depended on how loud the file was.
* ``resample`` is now a round trip back to the original rate, so it measures
  robustness to sample-rate conversion rather than a decoder's reaction to an
  unexpected rate.
* ``frequency_filter`` compared FFT bin frequencies for exact float equality
  and so was very nearly a no-op; it is deprecated in favour of ``notch``.
* All stochastic attacks take a seed and are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from taf.attacks.acoustic import AcousticChannel, EchoAttack, Reverberation
from taf.attacks.amplitude import Clipping, DynamicRangeCompression, GainChange
from taf.attacks.base import Attack, AttackError, AttackToolUnavailableError, deprecated_alias
from taf.attacks.codec import CodecCompression
from taf.attacks.filtering import (
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
    MovingAverageSmoothing,
    NotchFilter,
)
from taf.attacks.noise import AdditivePinkNoise, AdditiveWhiteNoise, ImpulseNoise
from taf.attacks.quantization import BitDepthReduction
from taf.attacks.resampling import ResampleRoundTrip, SampleRateOffset
from taf.attacks.temporal import (
    Cropping,
    PitchShift,
    SampleDropout,
    SampleInsertionDeletion,
    SpeedChange,
    TimeShift,
    TimeStretch,
    ZeroPadding,
)
from taf.models.WavFile import WavFile


@dataclass
class CorruptedWavFile(WavFile):
    """A signal under attack, built up one transformation at a time."""

    def __init__(self, wav_file: WavFile):
        self.samples = wav_file.samples
        self.samplerate = wav_file.samplerate
        self.path = wav_file.path
        self.applied: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- plumbing

    def apply(self, attack: Attack) -> "CorruptedWavFile":
        """Run any attack from the package and record what it did."""
        result = attack.apply(self.samples, self.samplerate)
        self.samples = result.audio
        self.samplerate = result.sample_rate
        self.applied.append(result.metadata)
        return self

    @property
    def metadata(self) -> list[dict[str, Any]]:
        """Metadata for every attack applied so far, in order."""
        return list(self.applied)

    # ------------------------------------------------------------------ noise

    def additive_noise(self, snr_db: float = 20.0, seed: int | None = 0) -> "CorruptedWavFile":
        """Additive white Gaussian noise at a target SNR.

        Changed: this used to take ``std``, an absolute noise amplitude drawn
        from the global RNG. That made the attack's severity depend on the
        recording level and made runs irreproducible.
        """
        return self.apply(AdditiveWhiteNoise(snr_db=snr_db, seed=seed))

    def pink_noise(self, snr_db: float = 20.0, seed: int | None = 0) -> "CorruptedWavFile":
        return self.apply(AdditivePinkNoise(snr_db=snr_db, seed=seed))

    def impulse_noise(
        self, snr_db: float = 20.0, density: float = 0.001, seed: int | None = 0
    ) -> "CorruptedWavFile":
        return self.apply(ImpulseNoise(snr_db=snr_db, density=density, seed=seed))

    def flip_random_samples(
        self, fraction: float = 0.001, seed: int | None = 0
    ) -> "CorruptedWavFile":
        """Deprecated: sign-flipping random samples models no real channel.

        Use :meth:`impulse_noise`, which produces comparable sparse damage at
        a controlled SNR.
        """
        deprecated_alias("flip_random_samples", "impulse_noise")
        return self.impulse_noise(density=fraction, seed=seed)

    def cut_random_samples(
        self, fraction: float = 0.01, run_length: int = 20, seed: int | None = 0
    ) -> "CorruptedWavFile":
        """Deprecated alias of :meth:`dropout`."""
        deprecated_alias("cut_random_samples", "dropout")
        return self.dropout(fraction=fraction, run_length=run_length, seed=seed)

    def sample_suppression(
        self, fraction: float = 0.01, run_length: int = 20, seed: int | None = 0
    ) -> "CorruptedWavFile":
        """Deprecated alias of :meth:`dropout`."""
        deprecated_alias("sample_suppression", "dropout")
        return self.dropout(fraction=fraction, run_length=run_length, seed=seed)

    def dropout(
        self, fraction: float = 0.01, run_length: int = 20, seed: int | None = 0
    ) -> "CorruptedWavFile":
        return self.apply(SampleDropout(fraction=fraction, run_length=run_length, seed=seed))

    # -------------------------------------------------------------- filtering

    def low_pass_filter(
        self, order: int = 6, cutoff_freq: float = 4000, zero_phase: bool = True
    ) -> "CorruptedWavFile":
        """Changed: order 6 rather than 16, and zero-phase by default.

        The cutoff is validated against Nyquist instead of being passed
        straight to the filter designer.
        """
        return self.apply(
            LowPassFilter(cutoff_hz=cutoff_freq, order=order, zero_phase=zero_phase)
        )

    def high_pass_filter(
        self, order: int = 6, cutoff_freq: float = 300, zero_phase: bool = True
    ) -> "CorruptedWavFile":
        return self.apply(
            HighPassFilter(cutoff_hz=cutoff_freq, order=order, zero_phase=zero_phase)
        )

    def band_pass_filter(
        self, low_hz: float = 300.0, high_hz: float = 3400.0, order: int = 6
    ) -> "CorruptedWavFile":
        return self.apply(BandPassFilter(low_hz=low_hz, high_hz=high_hz, order=order))

    def notch_filter(
        self, center_hz: float = 1000.0, quality: float = 30.0, depth_db: float | None = None
    ) -> "CorruptedWavFile":
        return self.apply(
            NotchFilter(center_hz=center_hz, quality=quality, depth_db=depth_db)
        )

    def frequency_filter(self, cutoff_frequency: float = 1000.0) -> "CorruptedWavFile":
        """Deprecated: the original removed FFT bins by exact float equality.

        ``np.abs(W) == cutoff_frequency`` matches only if the frequency grid
        happens to contain that exact value, so the attack usually did
        nothing at all and occasionally removed a single bin. Use
        :meth:`notch_filter`, which removes a defined band.
        """
        deprecated_alias("frequency_filter", "notch_filter")
        return self.notch_filter(center_hz=cutoff_frequency)

    def smoothing(self, window_length: int = 5) -> "CorruptedWavFile":
        return self.apply(MovingAverageSmoothing(window_length=window_length))

    # ------------------------------------------------------------- resampling

    def resample(self, target_samplerate: int = 8000) -> "CorruptedWavFile":
        """Round trip through ``target_samplerate`` and back.

        Changed: this used to leave the signal at the new rate, which meant
        the decoder was handed audio at a rate it was not built for. What that
        measured was API tolerance, not robustness to resampling.
        """
        return self.apply(ResampleRoundTrip(intermediate_hz=target_samplerate))

    def clock_drift(self, offset_ppm: float = 100.0) -> "CorruptedWavFile":
        return self.apply(SampleRateOffset(offset_ppm=offset_ppm))

    # ----------------------------------------------------------- quantization

    def quantization(self, bit_depth: int = 8, dither: bool = False) -> "CorruptedWavFile":
        return self.apply(BitDepthReduction(bits=bit_depth, dither=dither))

    # -------------------------------------------------------------- amplitude

    def amplitude_scaling(self, scale: float = 1.1) -> "CorruptedWavFile":
        """Scale by a linear factor, kept for compatibility.

        Prefer :meth:`gain`, which takes decibels - the unit a volume control
        actually works in and the one the literature reports.
        """
        if scale <= 0:
            raise AttackError(f"scale must be positive, got {scale}")
        return self.apply(GainChange(gain_db=float(20.0 * np.log10(scale))))

    def gain(self, gain_db: float = -6.0, prevent_clipping: bool = False) -> "CorruptedWavFile":
        return self.apply(GainChange(gain_db=gain_db, prevent_clipping=prevent_clipping))

    def clipping(self, threshold: float = 0.9, mode: str = "peak") -> "CorruptedWavFile":
        return self.apply(Clipping(threshold=threshold, mode=mode))

    def dynamic_range_compression(
        self, threshold: float = 0.3, ratio: float = 4.0
    ) -> "CorruptedWavFile":
        return self.apply(DynamicRangeCompression(threshold=threshold, ratio=ratio))

    # --------------------------------------------------------------- temporal

    def time_shift(self, shift_ms: float = 10.0, mode: str = "pad") -> "CorruptedWavFile":
        return self.apply(TimeShift(shift_ms=shift_ms, mode=mode))

    def crop(self, fraction: float = 0.01, position: str = "start") -> "CorruptedWavFile":
        """Changed: the cut is taken from one end by default and reported.

        The previous version always split the removal between both ends, which
        is a different attack and was not stated anywhere.
        """
        return self.apply(Cropping(fraction=fraction, position=position))

    def zero_padding(self, fraction: float = 0.1, position: str = "start") -> "CorruptedWavFile":
        return self.apply(ZeroPadding(fraction=fraction, position=position))

    def sample_jitter(
        self, events: int = 10, run_length: int = 8, operation: str = "delete", seed: int | None = 0
    ) -> "CorruptedWavFile":
        return self.apply(
            SampleInsertionDeletion(
                events=events, run_length=run_length, operation=operation, seed=seed
            )
        )

    def time_stretch(self, rate: float = 1.01) -> "CorruptedWavFile":
        """Changed: the default is a 1% change, not a doubling of tempo.

        A rate of 2.0 destroys every method for the trivial reason that half
        the signal is gone; the informative range is a few percent.
        """
        return self.apply(TimeStretch(rate=rate))

    def speed_change(self, rate: float = 1.01) -> "CorruptedWavFile":
        return self.apply(SpeedChange(rate=rate))

    def pitch_shift(self, n_steps: float = 0.5, bins_per_octave: int = 12) -> "CorruptedWavFile":
        """Changed: the default is half a semitone rather than four semitones.

        ``bins_per_octave`` is accepted for compatibility and converted to
        semitones, which is how the attack is now parameterised.
        """
        semitones = float(n_steps) * 12.0 / float(bins_per_octave)
        return self.apply(PitchShift(semitones=semitones))

    # --------------------------------------------------------------- acoustic

    def echo_addition(
        self, delay_seconds: float = 0.025, decay: float = 0.25
    ) -> "CorruptedWavFile":
        return self.apply(
            EchoAttack(delay_ms=delay_seconds * 1000.0, attenuation=decay)
        )

    def reverberation(self, rt60_seconds: float = 0.5, mix: float = 1.0) -> "CorruptedWavFile":
        return self.apply(Reverberation(rt60_seconds=rt60_seconds, mix=mix))

    def acoustic_channel(self, snr_db: float = 25.0, rt60_seconds: float = 0.3) -> "CorruptedWavFile":
        return self.apply(AcousticChannel(snr_db=snr_db, rt60_seconds=rt60_seconds))

    # ------------------------------------------------------------------ codec

    def mp3_compression(self, bitrate_kbps: int = 128) -> "CorruptedWavFile":
        return self.apply(CodecCompression(codec="mp3", bitrate_kbps=bitrate_kbps))

    def aac_compression(self, bitrate_kbps: int = 128) -> "CorruptedWavFile":
        return self.apply(CodecCompression(codec="aac", bitrate_kbps=bitrate_kbps))

    def opus_compression(self, bitrate_kbps: int = 64) -> "CorruptedWavFile":
        return self.apply(CodecCompression(codec="opus", bitrate_kbps=bitrate_kbps))

    def vorbis_compression(self, bitrate_kbps: int = 64) -> "CorruptedWavFile":
        return self.apply(CodecCompression(codec="vorbis", bitrate_kbps=bitrate_kbps))


__all__ = ["AttackToolUnavailableError", "CorruptedWavFile"]
