from typing import List

import numpy as np
from numpy.linalg import svd
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod


class FsvcMethod(SteganographyMethod):

    def __init__(self, sr: int, delta: float = 3.5793656):
        self.sr = sr
        self.d0 = 0.83
        self.delta = delta
        self.D = 0.65
        self.gamma1 = 0.136  # for 44.1 kHz sampling rate
        self.gamma2 = 0.181  # for 44.1 kHz sampling rate
        self.alpha = 3

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        gamma1 = self.gamma1 * self.sr / 44100
        gamma2 = self.gamma2 * self.sr / 44100

        frames = np.array_split(data, len(message))

        watermarked_frames = []

        for i, frame in enumerate(frames):
            x1, x2 = np.array_split(frame, 2)

            X1 = dct(x1, type=2, norm='ortho')
            X2 = dct(x2, type=2, norm='ortho')

            low = int(gamma1 * len(frame))
            high = int(gamma2 * len(frame) + 1)

            X1p = np.expand_dims(X1[low:high], axis=-1)
            X2p = np.expand_dims(X2[low:high], axis=-1)

            u1, s1, v1 = svd(X1p, full_matrices=False)
            u2, s2, v2 = svd(X2p, full_matrices=False)

            l1 = s1[0]
            l2 = s2[0]

            l1p, l2p = self.modify_svd_pair(self.alpha, l1, l2, message[i])

            s1p = np.array([l1p])
            s2p = np.array([l2p])

            X1p_em = u1 @ np.diag(s1p) @ v1
            X2p_em = u2 @ np.diag(s2p) @ v2

            X1p_em = np.squeeze(X1p_em)
            X2p_em = np.squeeze(X2p_em)

            X1[low:high] = X1p_em
            X2[low:high] = X2p_em

            x1_em = idct(X1, type=2, norm='ortho')
            x2_em = idct(X2, type=2, norm='ortho')

            frame_w = np.concatenate([x1_em, x2_em])

            watermarked_frames.append(frame_w)

        return np.concatenate(watermarked_frames)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        gamma1 = self.gamma1 * self.sr / 44100
        gamma2 = self.gamma2 * self.sr / 44100

        frames = np.array_split(data_with_watermark, watermark_length)
        watermark_bits = []

        for frame in frames:
            x1, x2 = np.array_split(frame, 2)

            X1 = dct(x1, type=2, norm='ortho')
            X2 = dct(x2, type=2, norm='ortho')

            low = int(gamma1 * len(frame))
            high = int(gamma2 * len(frame) + 1)

            X1p = np.expand_dims(X1[low:high], axis=-1)
            X2p = np.expand_dims(X2[low:high], axis=-1)

            u1, s1, v1 = svd(X1p, full_matrices=False)
            u2, s2, v2 = svd(X2p, full_matrices=False)

            l1 = s1[0]
            l2 = s2[0]

            if l1 / l2 < 1:
                watermark_bits.append(0)
            else:
                watermark_bits.append(1)

        return watermark_bits

    def type(self) -> str:
        return "Frequency Singular Value Coefficient Modification (FSVC) method"

    @staticmethod
    def modify_svd_pair(alpha, l1, l2, watermark_bit):
        if watermark_bit == 0:
            if l1 / l2 > 1 / (1 + alpha):
                l1 = (l1 + l2 * (1 + alpha)) / (alpha ** 2 + 2 * alpha + 2)
                l2 = (1 + alpha) * l1

        else:
            if l1 / l2 < 1 + alpha:
                l2 = (l2 + l1 * (1 + alpha)) / (alpha ** 2 + 2 * alpha + 2)
                l1 = (1 + alpha) * l2

        return l1, l2
