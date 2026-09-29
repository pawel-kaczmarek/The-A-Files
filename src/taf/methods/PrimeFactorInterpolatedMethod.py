"""Interpolation-based steganography with prime-factor variable capacity.

Every second sample is replaced by the interpolation of its two neighbours
plus a payload value, and the number of payload bits a position carries is
derived from the least prime factor of the log-scaled neighbour difference.
Both sides compute that capacity from the retained neighbours alone, so the
receiver is blind.

The previous implementation was written for integer PCM but was fed float
audio in [-1, 1], where log2(diff) is negative and the payload was added as
whole units, wrecking the signal (SNR -23 dB). It also returned twice as many
samples as the cover, and its decoder recovered the payload from a difference
that never held it (BER 27-56% even on integer input).
"""
import math
from typing import List

import numpy as np

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text

_INT16_MAX = 32767
_INT16_MIN = -32768


def _least_prime_factor(n: int) -> int:
    """Find the least prime factor of a number."""
    if n < 2:
        return n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


def _to_int16(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the signal as int16 samples plus whether it was float."""
    if np.issubdtype(audio.dtype, np.integer):
        return audio.astype(np.int64), False
    return np.clip(np.rint(audio.astype(np.float64) * _INT16_MAX),
                   _INT16_MIN, _INT16_MAX).astype(np.int64), True


class PrimeFactorInterpolatedMethod(SteganographyMethod):
    """
    Args:
        max_bits_per_sample: Upper bound on the payload width of one position.
            The least prime factor of a prime N is N itself, which would add a
            five-digit offset to a single sample; capping it keeps the worst
            case audible-but-small.
    """

    card = MethodCard(
        title=Text("Prime-factor interpolation", "Interpolacja z czynnikiem pierwszym"),
        abbreviation="PFI",
        family="lsb",
        purpose="steganography",
        references=(Reference("Adhiyaksa et al.", 2022, doi="10.1109/ISMODE53584.2022.9743066"),),
        summary=Text(
            en="Stores a variable number of bits as small offsets from interpolated sample values.",
            pl=(
                "Zapisuje zmienną liczbę bitów jako niewielkie odchylenia od interpolowanych wartości "
                "próbek."
            ),
        ),
        details=Text(
            en=(
                "Every second sample is replaced by an interpolation of its retained neighbours plus "
                "the payload value. Capacity follows the least prime factor of a log-scaled neighbour "
                "difference, capped by max_bits_per_sample. The decoder derives the same prediction and "
                "capacity from the neighbours. More bits mean larger offsets; changing the neighbours "
                "can corrupt decoding. Recovering the original cover is not part of the TAF interface, "
                "so reversibility is not established here."
            ),
            pl=(
                "Co druga próbka jest zastępowana interpolacją zachowanych sąsiadów powiększoną o "
                "wartość danych. Pojemność wynika z najmniejszego czynnika pierwszego logarytmicznie "
                "przeskalowanej różnicy sąsiadów i limitu max_bits_per_sample. Dekoder odtwarza tę samą "
                "predykcję oraz pojemność. Więcej bitów oznacza większe zmiany; modyfikacja sąsiadów "
                "może uszkodzić odczyt. Odtworzenie oryginalnego nośnika nie jest częścią interfejsu "
                "TAF, więc odwracalność nie jest tu wykazana."
            ),
        ),
    )

    def __init__(self, max_bits_per_sample: int = 4):
        if not 1 <= max_bits_per_sample <= 8:
            raise ValueError("max_bits_per_sample must be in [1, 8]")
        self.max_bits_per_sample = max_bits_per_sample

    def _capacity_at(self, left: int, right: int) -> int:
        """Bits carried by the position between two retained neighbours."""
        diff = abs(int(right) - int(left))
        n = int(math.floor(math.log2(diff))) if diff > 0 else 0
        width = _least_prime_factor(n) if n >= 2 else 1
        return int(min(max(width, 1), self.max_bits_per_sample))

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """Embed the message into the odd-indexed samples."""
        samples, was_float = _to_int16(data)
        stego = samples.copy()

        bits = [int(bit) for bit in message]
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("message must contain only 0 and 1 bits")

        bit_index = 0
        for position in range(1, len(samples) - 1, 2):
            if bit_index >= len(bits):
                break

            left, right = int(samples[position - 1]), int(samples[position + 1])
            width = self._capacity_at(left, right)
            chunk = bits[bit_index:bit_index + width]
            # A short trailing chunk is left-aligned, exactly as the decoder
            # reads it back.
            payload = 0
            for bit in chunk:
                payload = (payload << 1) | bit
            payload <<= width - len(chunk)
            bit_index += len(chunk)

            interpolated = (left + right) // 2
            stego[position] = np.clip(interpolated + payload, _INT16_MIN, _INT16_MAX)

        if bit_index < len(bits):
            raise CapacityError(
                f"message too long for cover audio: {len(bits)} > {bit_index} bits"
            )

        if was_float:
            return (stego.astype(np.float64) / _INT16_MAX).astype(data.dtype, copy=False)
        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """Recover the message from the odd-indexed samples."""
        samples, _ = _to_int16(data_with_watermark)
        bits: List[int] = []

        for position in range(1, len(samples) - 1, 2):
            if len(bits) >= watermark_length:
                break

            left, right = int(samples[position - 1]), int(samples[position + 1])
            width = self._capacity_at(left, right)
            payload = int(samples[position]) - (left + right) // 2
            payload = max(0, min(payload, (1 << width) - 1))

            for shift in range(width - 1, -1, -1):
                bits.append((payload >> shift) & 1)
                if len(bits) >= watermark_length:
                    break

        return bits[:watermark_length]

    def type(self) -> str:
        """Return the name of the steganography method."""
        return "Prime Factor Interpolated method"
