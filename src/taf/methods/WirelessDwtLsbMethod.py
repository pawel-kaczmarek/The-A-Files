"""DWT/LSB audio steganography for wireless-channel transmission experiments.

Reference:
    Hamdi, A. A., Eyssa, A. A., Abdalla, M. I., ElAffendi, M.,
    AlQahtani, A. A. S., Ateya, A. A., & Elsayed, R. A. (2025).
    "Improving Audio Steganography Transmission over Various Wireless
    Channels." Journal of Sensor and Actuator Networks, 14(6), 106.
    https://doi.org/10.3390/jsan14060106

The paper's second model embeds multimedia payload bits in the 8 LSBs of the
low-frequency DWT coefficients of an audio cover. This implementation adapts
that carrier-domain idea to the repository interface by embedding the provided
message bits directly, without the paper's image/audio payload staging.
"""
from __future__ import annotations

from math import ceil
from typing import List

import numpy as np
import pywt

from taf.models.SteganographyMethod import SteganographyMethod


def _pack_bits(bits: List[int], width: int) -> np.ndarray:
    packed = np.zeros(ceil(len(bits) / width), dtype=np.int64)

    for value in bits:
        if value not in (0, 1):
            raise ValueError("message must contain only 0 and 1 bits")

    for out_idx, bit_idx in enumerate(range(0, len(bits), width)):
        value = 0
        for bit in bits[bit_idx:bit_idx + width]:
            value = (value << 1) | int(bit)
        packed[out_idx] = value << (width - min(width, len(bits) - bit_idx))

    return packed


def _unpack_bits(values: np.ndarray, width: int, bit_count: int) -> List[int]:
    bits: List[int] = []

    for value in values:
        int_value = int(value)
        for shift in range(width - 1, -1, -1):
            bits.append((int_value >> shift) & 1)
            if len(bits) == bit_count:
                return bits

    return bits


def _to_float_audio(data: np.ndarray) -> tuple[np.ndarray, int | None]:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        peak = max(abs(info.min), info.max)
        return data.astype(np.float64) / peak, peak
    return data.astype(np.float64, copy=False), None


def _restore_dtype(data: np.ndarray, dtype: np.dtype, integer_peak: int | None) -> np.ndarray:
    if integer_peak is None:
        return data.astype(dtype, copy=False)

    info = np.iinfo(dtype)
    restored = np.rint(np.clip(data, -1.0, 1.0) * integer_peak)
    return np.clip(restored, info.min, info.max).astype(dtype)


class WirelessDwtLsbMethod(SteganographyMethod):
    """Embed message bits in low-frequency DWT coefficients using LSB coding."""

    def __init__(
        self,
        dwt_type: str = "haar",
        level: int = 1,
        lsb_depth: int = 8,
        coefficient_scale: int = 22000,
    ):
        if level < 1:
            raise ValueError("level must be positive")
        if not 1 <= lsb_depth <= 8:
            raise ValueError("lsb_depth must be in [1, 8]")
        if coefficient_scale <= 0:
            raise ValueError("coefficient_scale must be positive")

        self.dwt_type = dwt_type
        self.level = level
        self.lsb_depth = lsb_depth
        self.coefficient_scale = coefficient_scale

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        if len(message) == 0:
            return data.copy()

        audio, integer_peak = _to_float_audio(data)
        coeffs = pywt.wavedec(audio, self.dwt_type, mode="periodization", level=self.level)
        ll_band = coeffs[0]
        packed = _pack_bits(message, self.lsb_depth)

        if len(packed) > len(ll_band):
            capacity = len(ll_band) * self.lsb_depth
            raise ValueError(f"message too long for cover audio: {len(message)} > {capacity} bits")

        mask = (1 << self.lsb_depth) - 1
        quantized = np.rint(ll_band * self.coefficient_scale).astype(np.int64)
        quantized[:len(packed)] = (quantized[:len(packed)] & ~mask) | packed

        coeffs[0] = quantized.astype(np.float64) / self.coefficient_scale
        stego = pywt.waverec(coeffs, self.dwt_type, mode="periodization")
        stego = stego[:len(audio)]
        if len(stego) < len(audio):
            stego = np.pad(stego, (0, len(audio) - len(stego)))

        return _restore_dtype(stego, data.dtype, integer_peak)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        if watermark_length <= 0:
            return []

        audio, _ = _to_float_audio(data_with_watermark)
        coeffs = pywt.wavedec(audio, self.dwt_type, mode="periodization", level=self.level)
        ll_band = coeffs[0]
        packed_count = ceil(watermark_length / self.lsb_depth)

        if packed_count > len(ll_band):
            capacity = len(ll_band) * self.lsb_depth
            raise ValueError(f"watermark too long for stego audio: {watermark_length} > {capacity} bits")

        mask = (1 << self.lsb_depth) - 1
        quantized = np.rint(ll_band[:packed_count] * self.coefficient_scale).astype(np.int64)
        packed = quantized & mask
        return _unpack_bits(packed, self.lsb_depth, watermark_length)

    def type(self) -> str:
        return "Wireless DWT LSB method"
