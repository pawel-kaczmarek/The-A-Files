from typing import List
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import lfilter
from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography_methods.common.mixer import mixer


class EchoMethod(SteganographyMethod):
    """
    Implements the Echo Hiding watermarking technique using single echo kernels.
    This method embeds watermark bits by applying echoes with different delays to an audio signal.
    """

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Embed a watermark message into an audio signal using echo hiding.

        Args:
            data (np.ndarray): Input audio signal.
            message (List[int]): Watermark message bits to embed.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        d0, d1 = 150, 200  # Echo delays for bits 0 and 1
        alpha = 0.5        # Echo amplitude
        L = 8 * 1024       # Frame length

        nframe = np.floor(data.shape[0] / L)
        N = int(nframe - np.mod(nframe, 8))  # Number of usable frames for embedding

        # Adjust or truncate the message length to fit into the available frames
        bits = (message + N * [0])[:N] if len(message) < N else message[:N]

        # Create echo kernels for bits 0 and 1
        k0 = np.append(np.zeros(d0), [1]) * alpha
        k1 = np.append(np.zeros(d1), [1]) * alpha

        # Apply echoes to the signal
        echo_zero = lfilter(k0, 1, data)
        echo_one = lfilter(k1, 1, data)

        # Generate mixing window for embedding
        window = mixer(L, bits, 0, 1, 256)[0]

        # Embed watermark into the signal
        watermarked = (
            data[:N * L]
            + echo_zero[:N * L] * np.abs(window - 1)
            + echo_one[:N * L] * window
        )

        # Append the untouched part of the signal
        return np.append(watermarked, data[N * L:])

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        Extract a watermark message from a watermarked audio signal.

        Args:
            data_with_watermark (np.ndarray): Watermarked audio signal.
            watermark_length (int): Number of watermark bits to extract.

        Returns:
            List[int]: Extracted watermark bits.
        """
        d0, d1 = 150, 200  # Echo delays for bits 0 and 1
        L = 8 * 1024       # Frame length

        N = int(np.floor(len(data_with_watermark) / L))
        xsig = np.reshape(data_with_watermark[:N * L], (N, L)).T

        extracted_bits = []
        for k in range(N):
            rceps = np.real(ifft(np.log(np.abs(fft(xsig[:, k]) + 1e-10))))  # Add small constant to avoid log(0)
            extracted_bits.append(0 if rceps[d0] >= rceps[d1] else 1)

        return extracted_bits[:watermark_length]

    def type(self) -> str:
        """
        Return the type of the watermarking method.

        Returns:
            str: Method description.
        """
        return "Echo Hiding technique with single echo kernel"
