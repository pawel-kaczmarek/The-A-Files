"""Quantization Index Modulation with spread-transform dither modulation.

Reference:
    Chen, B., & Wornell, G. W. (2001). "Quantization index modulation: a class
    of provably good methods for digital watermarking and information
    embedding." IEEE Transactions on Information Theory, 47(4), 1423-1443.
    https://doi.org/10.1109/18.923725

Each bit is carried by the projection of one frame onto a key-derived
direction: the projection is quantised with a bit-dependent dither, and the
receiver picks the bit whose quantiser the projection sits closest to. Only
the component along the projection direction is modified, which is what makes
spread-transform dither modulation far less audible than quantising samples
directly.

The quantisation step is derived from the energy of the component orthogonal
to the projection direction. Embedding leaves that component untouched, so
both sides compute the same step, and because it scales with the signal the
watermark survives a volume change - unlike a fixed step, which is the usual
failure mode of QIM implementations.
"""
from typing import List

import numpy as np

from taf.models.SteganographyMethod import SteganographyMethod


class QimMethod(SteganographyMethod):
    """Spread-transform dither modulation (ST-DM)."""

    def __init__(self, key: int = 20240521, step_scale: float = 0.1, min_frame_length: int = 256):
        """
        Args:
            key: Seed shared by both sides; selects the projection directions
                and the dither values.
            step_scale: Quantisation step, relative to the norm of the
                orthogonal component of the frame. Larger steps are more
                robust and more audible.
            min_frame_length: Shortest usable frame; sets the capacity.
        """
        if step_scale <= 0:
            raise ValueError("step_scale must be positive")
        if min_frame_length < 2:
            raise ValueError("min_frame_length must be at least 2")

        self.key = key
        self.step_scale = step_scale
        self.min_frame_length = min_frame_length

    def _frame_length(self, sample_count: int, bit_count: int) -> int:
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        frame_length = sample_count // bit_count
        if frame_length < self.min_frame_length:
            capacity = sample_count // self.min_frame_length
            raise ValueError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return frame_length

    def _direction(self, frame_length: int, frame_index: int) -> np.ndarray:
        """Unit-norm projection direction for one frame."""
        rng = np.random.default_rng((self.key, frame_index))
        direction = rng.integers(0, 2, frame_length).astype(np.float64) * 2.0 - 1.0
        return direction / np.linalg.norm(direction)

    @staticmethod
    def _quantize(value: float, step: float, dither: float) -> float:
        return np.round((value - dither) / step) * step + dither

    def _dithers(self, step: float) -> tuple:
        """Interleaved quantisers, half a step apart, one per bit value."""
        return -step / 4.0, step / 4.0

    def _frame_state(self, frame: np.ndarray, frame_index: int) -> tuple:
        direction = self._direction(len(frame), frame_index)
        projection = float(np.dot(frame, direction))

        # Energy off the projection axis is untouched by embedding, so the
        # decoder derives exactly the same step as the encoder.
        orthogonal_energy = max(float(np.dot(frame, frame)) - projection ** 2, 0.0)
        step = self.step_scale * np.sqrt(orthogonal_energy)
        if step == 0.0:
            step = self.step_scale
        return direction, projection, float(step)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        frame_length = self._frame_length(len(audio), len(message))
        stego = audio.copy()

        for index, bit in enumerate(message):
            start = index * frame_length
            frame = stego[start:start + frame_length]

            direction, projection, step = self._frame_state(frame, index)
            dither = self._dithers(step)[int(bit)]
            target = self._quantize(projection, step, dither)

            frame += (target - projection) * direction

        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        frame_length = self._frame_length(len(audio), watermark_length)

        bits: List[int] = []
        for index in range(watermark_length):
            start = index * frame_length
            frame = audio[start:start + frame_length]

            _, projection, step = self._frame_state(frame, index)
            dither_zero, dither_one = self._dithers(step)

            error_zero = abs(projection - self._quantize(projection, step, dither_zero))
            error_one = abs(projection - self._quantize(projection, step, dither_one))
            bits.append(0 if error_zero <= error_one else 1)

        return bits

    def type(self) -> str:
        return "Quantization index modulation (spread-transform dither modulation)"
