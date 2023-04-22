from typing import List

import numpy as np
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography.common.split import to_frames, to_samples


class DctDeltaLsbMethod(SteganographyMethod):

    def __init__(self, sr: int, frame_length_in_ms: int = 100, delta_value: int = 10):
        self.sr = sr
        self.frame_length_in_ms = frame_length_in_ms
        self.delta_value = delta_value

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        first_frames, last_frame = to_frames(data, self.sr, self.frame_length_in_ms)
        frames = first_frames.copy()
        for x in range(0, len(message)):
            coeffs = dct(frames[x], type=2, norm='ortho')
            v1, v2 = self.sub_vectors(coeffs)

            norm1, u1 = self.norm_calc(v1)
            norm2, u2 = self.norm_calc(v2)

            norm = (norm1 + norm2) / 2

            delta = self.delta_value
            norm1, norm2 = ((norm + delta), (norm - delta)) if message[x] == 1 else ((norm - delta), (norm + delta))

            v1 = self._inv_norm_calc(norm1, u1)
            v2 = self._inv_norm_calc(norm2, u2)

            frames[x] = idct(self.inv_sub_vectors(v1, v2), type=2, norm='ortho')
        return to_samples(frames, last_frame)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        decoded_watermark = list()
        first_frames, last_frame = to_frames(data_with_watermark, self.sr, self.frame_length_in_ms)
        frames = first_frames.copy()

        for x in range(0, watermark_length):
            coeffs = dct(frames[x], type=2, norm='ortho')
            coeffs_len = len(coeffs)
            if x < coeffs_len:
                v1, v2 = self.sub_vectors(coeffs)

                norm1, u1 = self.norm_calc(v1)
                norm2, u2 = self.norm_calc(v2)

                value = 1 if norm1 > norm2 else 0

                decoded_watermark.append(value)
            else:
                decoded_watermark.append(0)

        return decoded_watermark

    def type(self) -> str:
        return "DCT Delta LSB method"

    @staticmethod
    def sub_vectors(coeff):
        coeffs_len = len(coeff) // 2

        v1 = coeff[coeffs_len:]
        v2 = coeff[:coeffs_len]

        return v1, v2

    @staticmethod
    def inv_sub_vectors(v1, v2):
        return np.concatenate((v2, v1))

    @staticmethod
    def norm_calc(v):
        norm = 0
        for coeff in v:
            norm += coeff ** 2
        if type(norm) != float:
            norm = np.sqrt(norm.astype(float))
        else:
            norm = np.sqrt(norm)
        u = v / norm
        return norm, u

    @staticmethod
    def _inv_norm_calc(norm, u):
        vLen = len(u)
        v = np.zeros(vLen)

        for i in range(vLen):
            v[i] = norm * u[i]

        return v
