from typing import List

import numpy as np
import pywt

from TAF.models.SteganographyMethod import SteganographyMethod


class DwtLsbMethod(SteganographyMethod):

    def __init__(self, dwt_type: str = 'bior5.5'):
        self.dwt_type = dwt_type

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        coeffs = pywt.wavedec(data, self.dwt_type, mode='sym', level=2)  # DWT
        cA2, cD2, cD1 = coeffs

        for i in range(0, len(message)):
            cD2[10 * (i + 1)] = message[i]  # 10 is not important

        coeffs = cA2, cD2, cD1
        return pywt.waverec(coeffs, self.dwt_type, mode='sym')

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        coeffs = pywt.wavedec(data_with_watermark, self.dwt_type, mode='sym', level=2)  # DWT
        cA2, cD2, cD1 = coeffs

        return [
            int(np.rint(cD2[10 * (x + 1)])) for x in range(watermark_length)
        ]

    def type(self) -> str:
        return "DWT LSB based method"
