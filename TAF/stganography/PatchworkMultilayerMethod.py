from typing import List

import numpy as np
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod


class PatchworkMultilayerMethod(SteganographyMethod):

    def __init__(self, sr: int):
        self.sr = sr
        self.fs = 3000  # starting frequency for watermark embedding
        self.fe = 7000  # ending frequency for watermark embedding
        self.k1 = 0.195
        self.k2 = 0.08

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        L = len(data)

        si = int(self.fs / (self.sr / L))
        ei = int(self.fe / (self.sr / L))

        X = dct(data, type=2, norm='ortho')

        Xs = X[si:(ei + 1)]
        Ls = len(Xs)

        if Ls % (len(message) * 2) != 0:
            Ls -= Ls % (len(message) * 2)
            Xs = Xs[:Ls]

        Xsp = np.dstack((Xs[:Ls // 2], Xs[:(Ls // 2 - 1):-1])).flatten()

        # only the first layer
        segments = np.array_split(Xsp, len(message) * 2)
        watermarked_segments = []
        for i in range(0, len(segments), 2):

            j = i // 2 + 1
            rj = self.k1 * np.exp(-self.k2 * j)

            m1j = np.mean(np.abs(segments[i]))
            m2j = np.mean(np.abs(segments[i + 1]))

            mj = (m1j + m2j) / 2
            mmj = min(m1j, m2j)

            m1jp = m1j
            m2jp = m2j

            if message[j - 1] == 0 and (m1j - m2j) < rj * mmj:
                m1jp = mj + (rj * mmj / 2)
                m2jp = mj - (rj * mmj / 2)
            elif message[j - 1] == 1 and (m2j - m1j) < rj * mmj:
                m1jp = mj - (rj * mmj / 2)
                m2jp = mj + (rj * mmj / 2)

            Y1j = segments[i] * m1jp / m1j
            Y2j = segments[i + 1] * m2jp / m2j

            watermarked_segments.append(Y1j)
            watermarked_segments.append(Y2j)

        Ysp = np.hstack(watermarked_segments)
        Ys = np.hstack([Ysp[::2], Ysp[-1::-2]])

        Y = X[:]
        Y[si:(si + Ls)] = Ys
        watermarked_data = idct(Y, type=2, norm='ortho')

        return watermarked_data

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        L = len(data_with_watermark)

        si = int(self.fs / (self.sr / L))
        ei = int(self.fe / (self.sr / L))

        X = dct(data_with_watermark, type=2, norm='ortho')

        Xs = X[si:(ei + 1)]
        Ls = len(Xs)

        if Ls % (watermark_length * 2) != 0:
            Ls -= Ls % (watermark_length * 2)
            Xs = Xs[:Ls]

        Xsp = np.dstack((Xs[:Ls // 2], Xs[:(Ls // 2 - 1):-1])).flatten()

        segments = np.array_split(Xsp, watermark_length * 2)
        watermark_bits = []

        for i in range(0, len(segments), 2):

            j = i // 2 + 1
            rj = self.k1 * np.exp(-self.k2 * j)

            m1j = np.mean(np.abs(segments[i]))
            m2j = np.mean(np.abs(segments[i + 1]))

            dj = m1j - m2j

            if dj >= 0:
                watermark_bits.append(0)
            else:
                watermark_bits.append(1)

        return watermark_bits

    def type(self) -> str:
        return "Patchwork-Based multilayer audio method"
