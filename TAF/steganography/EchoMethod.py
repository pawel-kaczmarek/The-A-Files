from typing import List

import numpy as np
from loguru import logger
from scipy.fft import fft, ifft
from scipy.signal import lfilter

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography.common.mixer import mixer


class EchoMethod(SteganographyMethod):

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        d0 = 150  # Delay rate
        d1 = 200  # Delay rate for bit
        alpha = 0.5  # Echo amplitude
        L = 8 * 1024  # Length of frames

        bit = message
        nframe = np.floor(data.shape[0] / L)
        N = int(nframe - np.mod(nframe, 8))  # Number of frames( for 8 bit)
        if len(bit) > N:
            logger.debug('Message is too long, being cropped!')
            bits = bit[:N]
        else:
            logger.debug('Message is being zero padded...')
            bits = (bit + N * [0])[:N]

        k0 = np.append(np.zeros(d0), 1) * alpha  # Echo kernel for bit0
        k1 = np.append(np.zeros(d1), 1) * alpha  # Echo kernel for bit1

        echo_zro = lfilter(k0, 1, data)
        echo_one = lfilter(k1, 1, data)
        window = mixer(L, bits, 0, 1, 256)[0]
        out = data[0: N * L] + echo_zro[0: N * L] * np.abs(window - 1) + echo_one[0: N * L] * window
        out = np.append(out, data[N * L: len(data)])

        return out

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        d0 = 150  # Delay rate
        d1 = 200  # Delay rate for bit
        alpha = 0.5  # Echo amplitude
        L = 8 * 1024  # Length of frames
        N = int(np.floor(len(data_with_watermark) / L))
        xsig = np.reshape(np.transpose(data_with_watermark[0:N * L]), (L, N), order='F')
        data = np.empty(N)

        for k in range(N):
            rceps = ifft(np.log(np.abs(fft(xsig[:, k]))))
            if rceps[d0] >= rceps[d1]:
                data[k] = 0
            else:
                data[k] = 1

        return np.asarray(data, dtype=np.int)[:watermark_length]

    def type(self) -> str:
        return "Echo Hiding technique with single echo kernel"
