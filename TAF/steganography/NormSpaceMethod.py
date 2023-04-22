from typing import List

import numpy as np
import pywt
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography.common.split import to_frames


class NormSpaceMethod(SteganographyMethod):

    def __init__(self, sr: int, delta: float = 0.01):
        self.sr = sr
        self.delta = delta

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        segments, last_frame = to_frames(data, self.sr, len(data) / self.sr / len(message) * 1000)
        segments = segments.copy()
        rsegments = []

        for ind, segment in enumerate(segments):

            cA1, cD1 = pywt.dwt(segment, 'db1')

            v = dct(cA1, norm='ortho')

            v1 = v[::2]
            v2 = v[1::2]

            nrmv1 = np.linalg.norm(v1, ord=2)
            nrmv2 = np.linalg.norm(v2, ord=2)

            u1 = v1 / nrmv1
            u2 = v2 / nrmv2

            watermark_bit = message[ind]
            nrm = (nrmv1 + nrmv2) / 2
            if watermark_bit == 1:
                nrmv1 = nrm + self.delta
                nrmv2 = nrm - self.delta
            else:
                nrmv1 = nrm - self.delta
                nrmv2 = nrm + self.delta

            rv1 = nrmv1 * u1
            rv2 = nrmv2 * u2

            rv = np.zeros((len(v),))

            rv[::2] = rv1
            rv[1::2] = rv2

            rcA1 = idct(rv, norm='ortho')

            rseg = pywt.idwt(rcA1, cD1, 'db1')
            rsegments.append(rseg[:])

        if last_frame is not None:
            rsegments.append(last_frame)
        return np.concatenate(rsegments)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        segments, last_frame = to_frames(data_with_watermark, self.sr,
                                         len(data_with_watermark) / self.sr / watermark_length * 1000)
        segments = segments.copy()
        watermark_bits = []

        for ind, segment in enumerate(segments):
            cA1, cD1 = pywt.dwt(segment, 'db1')

            v = dct(cA1, norm='ortho')

            v1 = v[::2]
            v2 = v[1::2]

            nrmv1 = np.linalg.norm(v1, ord=2)
            nrmv2 = np.linalg.norm(v2, ord=2)

            if nrmv1 > nrmv2:
                watermark_bits.append(1)
            else:
                watermark_bits.append(0)

        return watermark_bits

    def type(self) -> str:
        return "Norm space method"
