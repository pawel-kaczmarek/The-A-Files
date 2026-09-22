"""Time-domain watermarking by low-frequency amplitude modification.

Reference:
    Lie, W.-N., & Chang, L.-C. (2006). "Robust and high-quality time-domain
    audio watermarking based on low-frequency amplitude modification." IEEE
    Transactions on Multimedia, 8(1), 46-59.
    https://doi.org/10.1109/TMM.2005.861292

Each bit is carried by the amplitude relation between three consecutive
sub-segments of the low-frequency component: the middle one is pushed below
or above the average of its neighbours. Because the payload lives in a
relation between energies rather than in individual sample values, extraction
needs no cover signal and is unaffected by a volume change, and because it
lives in the low band it survives low-pass filtering, resampling and additive
noise.

The relation is enforced by scaling whole sub-segments, which is a gradual
gain change rather than a per-sample edit - the reason the paper reports high
perceptual quality for a time-domain scheme.
"""
from typing import List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from taf.models.SteganographyMethod import SteganographyMethod


class LowFrequencyAmplitudeMethod(SteganographyMethod):
    """Low-frequency amplitude modification (LFAM)."""

    def __init__(self, sr: int = 16000, cutoff: float = 2000.0, margin: float = 0.3,
                 min_segment_length: int = 768):
        """
        Args:
            sr: Sampling rate, needed for the low-frequency split.
            cutoff: Upper edge of the low-frequency carrier band, in Hz.
            margin: Relative amplitude gap enforced between the middle
                sub-segment and the mean of its neighbours.
            min_segment_length: Shortest usable segment; sets the capacity.
        """
        if not 0 < cutoff < sr / 2:
            raise ValueError("cutoff must be between 0 and the Nyquist frequency")
        if margin <= 0:
            raise ValueError("margin must be positive")
        if min_segment_length < 3:
            raise ValueError("min_segment_length must be at least 3")

        self.sr = sr
        self.cutoff = cutoff
        self.margin = margin
        self.min_segment_length = min_segment_length

    def _segment_length(self, sample_count: int, bit_count: int) -> int:
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        segment_length = sample_count // bit_count
        if segment_length < self.min_segment_length:
            capacity = sample_count // self.min_segment_length
            raise ValueError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return segment_length

    def _split(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sos = butter(4, self.cutoff, fs=self.sr, btype="low", output="sos")
        low = sosfiltfilt(sos, audio)
        return low, audio - low

    @staticmethod
    def _thirds(segment_length: int) -> Tuple[slice, slice, slice]:
        third = segment_length // 3
        return slice(0, third), slice(third, 2 * third), slice(2 * third, 3 * third)

    @staticmethod
    def _amplitudes(segment: np.ndarray, parts: Tuple[slice, slice, slice]) -> Tuple[float, float, float]:
        return tuple(float(np.mean(np.abs(segment[part]))) for part in parts)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        segment_length = self._segment_length(len(audio), len(message))
        low, high_band = self._split(audio)
        carrier = low.copy()
        parts = self._thirds(segment_length)

        for index, bit in enumerate(message):
            start = index * segment_length
            segment = carrier[start:start + segment_length]
            first, middle, last = self._amplitudes(segment, parts)

            neighbours = (first + last) / 2.0
            if neighbours == 0.0 or middle == 0.0:
                continue

            # Bit 1 wants the middle sub-segment above its neighbours' mean,
            # bit 0 below, by the given relative margin. Only the shortfall is
            # corrected, and it is shared between the middle and its
            # neighbours in opposite directions: forcing the middle onto a
            # fixed target instead means rescaling a quiet passage to the
            # level of a loud one, which cost ~30 dB of SNR.
            ratio = middle / neighbours
            target_ratio = (1.0 + self.margin) if int(bit) == 1 else 1.0 / (1.0 + self.margin)

            if (int(bit) == 1 and ratio >= target_ratio) or (
                int(bit) == 0 and ratio <= target_ratio
            ):
                continue

            scale = np.sqrt(target_ratio / ratio)
            segment[parts[1]] *= scale
            segment[parts[0]] /= scale
            segment[parts[2]] /= scale

        # Band-limit the modification alone; filtering the carrier itself
        # would compound the filter roll-off into the stego signal.
        modification, _ = self._split(carrier - low)
        return (low + modification + high_band).astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        segment_length = self._segment_length(len(audio), watermark_length)
        low, _ = self._split(audio)
        parts = self._thirds(segment_length)

        bits: List[int] = []
        for index in range(watermark_length):
            start = index * segment_length
            segment = low[start:start + segment_length]
            first, middle, last = self._amplitudes(segment, parts)
            bits.append(1 if middle >= (first + last) / 2.0 else 0)

        return bits

    def type(self) -> str:
        return "Low-frequency amplitude modification (LFAM) method"
