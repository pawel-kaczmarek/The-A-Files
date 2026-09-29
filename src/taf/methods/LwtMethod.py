from typing import List

import numpy as np
import pywt

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text


class LwtMethod(SteganographyMethod):

    card = MethodCard(
        title="LWT",
        family="transform",
        purpose="watermarking",
        strength_parameter="threshold",
        references=(Reference("Mushtaq et al.", 2024, doi="10.1109/ICRITO61523.2024.10522195"),),
        summary=Text(
            en="Encodes bits as the sign of blocks of wavelet detail coefficients.",
            pl="Koduje bity jako znak bloków współczynników detali falkowych.",
        ),
        details=Text(
            en=(
                "This implementation uses a two-level Haar wavelet decomposition and groups eight "
                "detail coefficients per bit. It forces a positive or negative block with a minimum "
                "magnitude set by threshold, then reconstructs the signal. Decoding reads the sign of "
                "the block mean. A larger threshold creates a stronger mark and more distortion; "
                "alignment and detail-band preservation matter. The transform is the PyWavelets Haar "
                "decomposition rather than an explicit lifting scheme."
            ),
            pl=(
                "Ta implementacja używa dwupoziomowej dekompozycji Haara i grupuje po osiem "
                "współczynników detali na bit. Wymusza dodatni lub ujemny blok z minimalną wartością "
                "bezwzględną threshold, po czym odtwarza sygnał. Dekoder odczytuje znak średniej bloku. "
                "Większy próg wzmacnia znak i zniekształcenia; istotne są synchronizacja oraz "
                "zachowanie pasma detali. Transformacją jest dekompozycja Haara z PyWavelets, a nie "
                "jawny schemat liftingowy."
            ),
        ),
    )

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        data_flat = data.flatten()

        coeffs = pywt.wavedec(data_flat, 'haar', level=2)
        LH = coeffs[2]

        block_size = 8
        num_blocks = len(LH) // block_size

        message_flat = np.array(message).flatten()
        if len(message_flat) > num_blocks:
            raise CapacityError(
                f"message too long for cover audio: {len(message_flat)} > {num_blocks} bits"
            )

        for idx in range(min(num_blocks, len(message_flat))):
            block_start = idx * block_size
            block_end = block_start + block_size

            if message_flat[idx] == 1:
                LH[block_start:block_end] = np.maximum(0.25 * LH[block_start:block_end], self.threshold)
            else:
                LH[block_start:block_end] = np.minimum(-0.25 * LH[block_start:block_end], -self.threshold)

        coeffs[2] = LH
        watermarked_data_flat = pywt.waverec(coeffs, 'haar')
        # waverec pads odd-length signals, so trim back to the cover length.
        return watermarked_data_flat[:len(data_flat)]

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        data_flat = data_with_watermark.flatten()

        coeffs = pywt.wavedec(data_flat, 'haar', level=2)
        LH = coeffs[2]

        block_size = 8
        watermark_bits = []
        for idx in range(watermark_length):
            block_start = idx * block_size
            block_end = block_start + block_size

            if block_end > len(LH):
                break

            # The encoder forces a whole block to one sign, so the sign of the
            # block mean carries the bit. Comparing against the absolute
            # embedding threshold instead would break under any amplitude
            # scaling of the stego signal.
            mean_value = np.mean(LH[block_start:block_end])
            watermark_bits.append(1 if mean_value >= 0 else 0)

        return watermark_bits

    def type(self) -> str:
        return "LWT method"
