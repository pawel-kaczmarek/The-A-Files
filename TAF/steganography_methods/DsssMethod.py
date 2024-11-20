from typing import List
import numpy as np
from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography_methods.common.mixer import mixer


class DsssMethod(SteganographyMethod):
    """
    Implements the Direct Sequence Spread Spectrum (DSSS) technique for audio watermarking.

    This method encodes and decodes watermark bits into/from an audio signal using spread spectrum principles.
    """

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Encode a watermark message into an audio signal using DSSS.

        Args:
            data (np.ndarray): Input audio signal.
            message (List[int]): Watermark message bits to embed.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        L_min = 8 * 1024  # Minimum segment length
        L2 = np.floor(len(data) / len(message))  # Length of each segment
        L = int(max(L_min, L2))  # Ensure segment length is sufficiently large
        nframe = np.floor(len(data) / L)
        N = int(nframe - np.mod(nframe, 8))  # Ensure the number of segments is a multiple of 8

        alpha = 0.005  # Embedding strength

        # Prepare watermark bits
        bits = message[:N] if len(message) > N else (message + [0] * N)[:N]

        # Generate pseudo-random sequence
        r = np.ones((L,))  # Simulated with all ones (adjust PRNG as needed)
        pr = np.tile(r, N)

        # Mix watermark bits with the pseudo-random sequence
        mix = mixer(L, bits, -1, 1, 256)[0]
        stego = data[:N * L] + mix * pr * alpha  # Apply watermarking

        # Append unmodified part of the audio signal
        return np.concatenate((stego, data[N * L:]))

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        Decode a watermark message from a watermarked audio signal using DSSS.

        Args:
            data_with_watermark (np.ndarray): Watermarked audio signal.
            watermark_length (int): Length of the watermark message to extract.

        Returns:
            List[int]: Decoded watermark bits.
        """
        L_min = 8 * 1024  # Minimum segment length
        L2 = np.floor(len(data_with_watermark) / watermark_length)  # Length of each segment
        L = int(max(L_min, L2))  # Ensure segment length is sufficiently large
        nframe = np.floor(len(data_with_watermark) / L)
        N = int(nframe - np.mod(nframe, 8))  # Ensure the number of segments is a multiple of 8

        # Reshape the watermarked signal into segments
        xsig = data_with_watermark[:N * L].reshape(N, L).T
        r = np.ones((L,))  # Simulated with all ones (adjust PRNG as needed)

        # Decode watermark bits
        c = np.dot(r, xsig) / L  # Correlation
        decoded_bits = (c > 0).astype(int)  # Thresholding

        return decoded_bits[:watermark_length].tolist()

    def type(self) -> str:
        """Return the type of watermarking method."""
        return "Direct Sequence Spread Spectrum (DSSS) technique"
