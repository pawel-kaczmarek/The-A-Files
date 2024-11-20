from typing import List
import bitstring
import numpy as np
from TAF.models.SteganographyMethod import SteganographyMethod


class LsbMethod(SteganographyMethod):

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        for idx, m in enumerate(message):
            # Convert the floating-point number to a 32-bit binary string
            bit_array = bitstring.BitArray(float=data[idx], length=32).bin
            # Replace the least significant bit (LSB) with the message bit
            bit_array = bit_array[:-1] + str(m)
            # Convert the modified binary string back to a float and update the data array
            data[idx] = bitstring.BitArray(bin=bit_array, length=32).float
        return data

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        # Extract the LSB from each sample and reconstruct the message
        return [
            int(bitstring.BitArray(float=data_with_watermark[x], length=32).bin[-1]) for x in range(watermark_length)
        ]

    def type(self) -> str:
        return "Standard LSB coding method"
