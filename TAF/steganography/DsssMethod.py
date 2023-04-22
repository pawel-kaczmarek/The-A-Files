from typing import List

import numpy as np
from loguru import logger

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography.common.mixer import mixer


class DsssMethod(SteganographyMethod):

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:

        L_min = 8 * 1024  # Setting a minimum value for segment length
        L2 = np.floor(len(data) / len(message))  # Length of segments
        L = int(max(L_min, L2))  # Keeping length of segments big enough
        nframe = np.floor(len(data) / L)
        N = int(nframe - np.mod(nframe, 8))  # Number of segments(for 8 bits)

        alpha = 0.005

        bit = message
        if len(bit) > N:
            logger.debug('Message is too long, being cropped!')
            bits = bit[:N]
        else:
            logger.debug('Message is being zero padded...')
            bits = (bit + N * [0])[:N]

        r = np.ones((L, 1))  # r = prng('password', L);
        pr = np.reshape(r * np.ones((1, N)), (N * L))
        mix = mixer(L, bits, -1, 1, 256)[0]
        stego = np.add(data[0:N * L], np.multiply(mix, pr) * alpha)  # Using first channel
        out = np.append(stego, data[N * L: len(data)])
        return out

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        L_min = 8 * 1024  # Setting a minimum value for segment length
        L2 = np.floor(len(data_with_watermark) / watermark_length)  # Length of segments
        L = int(max(L_min, L2))  # Keeping length of segments big enough
        nframe = np.floor(len(data_with_watermark) / L)
        N = int(nframe - np.mod(nframe, 8))  # Number of segments(for 8 bits)

        xsig = np.reshape(np.transpose(data_with_watermark[0:N * L]), (L, N), order='F')
        r = np.ones((L, 1))  # r = prng('password', L);

        data = np.empty(N)
        c = np.zeros((N, 1))
        for k in range(N):
            c[k] = np.sum(xsig[:, k] * r) / L
            if c[k] < 0:
                data[k] = 0
            else:
                data[k] = 1

        return list(np.asarray(data, dtype=np.int)[:watermark_length])

    def type(self) -> str:
        return "Direct Sequence Spread Spectrum technique"
