from typing import List

import bitstring
import numpy as np

from TAF.models.SteganographyMethod import SteganographyMethod


class LsbMethod(SteganographyMethod):

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        for idx, m in enumerate(message):
            bit_array = bitstring.BitArray(float=data[idx], length=32).bin
            bit_array = bit_array[:-1] + str(m)
            data[idx] = bitstring.BitArray(bin=bit_array, length=32).float
        return data

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        return [
            int(bitstring.BitArray(float=data_with_watermark[x], length=32).bin[-1]) for x in range(watermark_length)
        ]

    def type(self) -> str:
        return "Standard LSB coding method"
