"""Improved spread spectrum (ISS) audio watermarking.

References:
    Malvar, H. S., & Florencio, D. A. F. (2003). "Improved spread spectrum: a
    new modulation technique for robust watermarking." IEEE Transactions on
    Signal Processing, 51(4), 898-905.
    https://doi.org/10.1109/TSP.2003.809385

    Cox, I. J., Kilian, J., Leighton, F. T., & Shamoon, T. (1997). "Secure
    spread spectrum watermarking for multimedia." IEEE Transactions on Image
    Processing, 6(12), 1673-1687. https://doi.org/10.1109/83.650120

Classic spread spectrum (Cox et al.) adds a bipolar chip sequence to the
carrier and detects it by correlation, which leaves the host signal itself as
the dominant noise source at the receiver. ISS removes that term at the
transmitter: the component of the carrier along the chip direction is
subtracted before the watermark is added, so the projection the receiver sees
is set by the embedder alone.

The repository's DSSS method is the plain-correlation variant; this one is
its host-interference-rejecting counterpart, and the pair is the interesting
comparison.
"""
from typing import List

import numpy as np

from taf.models.SteganographyMethod import SteganographyMethod


class ImprovedSpreadSpectrumMethod(SteganographyMethod):
    """Host-interference-rejecting spread spectrum watermarking."""

    def __init__(
        self,
        key: int = 20240521,
        strength: float = 0.05,
        removal: float = 1.0,
        min_chip_length: int = 256,
    ):
        """
        Args:
            key: Seed of the chip generator, shared by both sides.
            strength: Watermark amplitude, relative to the norm of the frame
                component orthogonal to the chip sequence.
            removal: Host-rejection factor (lambda in the paper). 1.0 removes
                the host projection completely; 0.0 degenerates to classic
                spread spectrum.
            min_chip_length: Shortest usable frame; sets the capacity.
        """
        if strength <= 0:
            raise ValueError("strength must be positive")
        if not 0.0 <= removal <= 1.0:
            raise ValueError("removal must be in [0, 1]")
        if min_chip_length < 2:
            raise ValueError("min_chip_length must be at least 2")

        self.key = key
        self.strength = strength
        self.removal = removal
        self.min_chip_length = min_chip_length

    def _frame_length(self, sample_count: int, bit_count: int) -> int:
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        frame_length = sample_count // bit_count
        if frame_length < self.min_chip_length:
            capacity = sample_count // self.min_chip_length
            raise ValueError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return frame_length

    def _chips(self, frame_length: int, frame_index: int) -> np.ndarray:
        """Unit-norm chip sequence for one frame."""
        rng = np.random.default_rng((self.key, frame_index))
        chips = rng.integers(0, 2, frame_length).astype(np.float64) * 2.0 - 1.0
        return chips / np.linalg.norm(chips)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        frame_length = self._frame_length(len(audio), len(message))
        stego = audio.copy()

        for index, bit in enumerate(message):
            start = index * frame_length
            frame = stego[start:start + frame_length]

            chips = self._chips(frame_length, index)
            projection = float(np.dot(frame, chips))

            # Amplitude is referenced to the energy that stays orthogonal to
            # the chips, so it is unaffected by the embedding itself.
            orthogonal_energy = max(float(np.dot(frame, frame)) - projection ** 2, 0.0)
            amplitude = self.strength * np.sqrt(orthogonal_energy)

            sign = 1.0 if int(bit) == 1 else -1.0
            frame += (sign * amplitude - self.removal * projection) * chips

        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        frame_length = self._frame_length(len(audio), watermark_length)

        bits: List[int] = []
        for index in range(watermark_length):
            start = index * frame_length
            frame = audio[start:start + frame_length]
            chips = self._chips(frame_length, index)
            bits.append(1 if float(np.dot(frame, chips)) >= 0.0 else 0)

        return bits

    def type(self) -> str:
        return "Improved spread spectrum (ISS) technique"
