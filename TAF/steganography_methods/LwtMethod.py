from typing import List

import numpy as np
import pywt

from TAF.models.SteganographyMethod import SteganographyMethod


class LwtMethod(SteganographyMethod):

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        data_flat = data.flatten()

        coeffs = pywt.wavedec(data_flat, 'haar', level=2)
        LH = coeffs[2]

        block_size = 8
        num_blocks = len(LH) // block_size

        message_flat = np.array(message).flatten()
        for idx in range(min(num_blocks, len(message_flat))):
            block_start = idx * block_size
            block_end = block_start + block_size

            if message_flat[idx] == 1:
                LH[block_start:block_end] = np.maximum(0.25 * LH[block_start:block_end], self.threshold)
            else:
                LH[block_start:block_end] = np.minimum(-0.25 * LH[block_start:block_end], -self.threshold)

        coeffs[2] = LH
        watermarked_data_flat = pywt.waverec(coeffs, 'haar')
        return watermarked_data_flat

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

            mean_value = np.mean(LH[block_start:block_end])
            watermark_bits.append(1 if mean_value > self.threshold else 0)

        return watermark_bits

    def type(self) -> str:
        return "LWT method"
