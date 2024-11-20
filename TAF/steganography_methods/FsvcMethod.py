from typing import List

import numpy as np
from numpy.linalg import svd
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod


def modify_svd_pair(alpha, l1, l2, watermark_bit):
    """
    Modify the singular values based on the watermark bit and a scaling factor.

    Args:
        alpha (float): Scaling factor.
        l1 (float): First singular value.
        l2 (float): Second singular value.
        watermark_bit (int): Watermark bit (0 or 1).

    Returns:
        tuple: Modified singular values.
    """
    if watermark_bit == 0:
        if l1 / l2 > 1 / (1 + alpha):
            l1 = (l1 + l2 * (1 + alpha)) / (alpha ** 2 + 2 * alpha + 2)
            l2 = (1 + alpha) * l1
    else:
        if l1 / l2 < 1 + alpha:
            l2 = (l2 + l1 * (1 + alpha)) / (alpha ** 2 + 2 * alpha + 2)
            l1 = (1 + alpha) * l2

    return l1, l2


class FsvcMethod(SteganographyMethod):

    def __init__(self, sr: int, delta: float = 3.5793656):
        self.sr = sr
        self.d0 = 0.83  # Constant for scaling the singular values
        self.delta = delta  # Delta for controlling embedding strength
        self.D = 0.65  # Another constant related to the frequency domain
        self.gamma1 = 0.136  # Scaling factor for lower frequency band
        self.gamma2 = 0.181  # Scaling factor for higher frequency band
        self.alpha = 3  # Influence of watermark bits on the singular values

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Embed the watermark message into the audio signal using SVD and DCT transformations.

        Args:
            data (np.ndarray): Input audio signal.
            message (List[int]): Watermark message bits to embed.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        gamma1 = self.gamma1 * self.sr / 44100
        gamma2 = self.gamma2 * self.sr / 44100

        # Split the audio data into frames based on the watermark message length
        frames = np.array_split(data, len(message))

        watermarked_frames = []

        # Loop through each frame and embed the watermark
        for i, frame in enumerate(frames):
            x1, x2 = np.array_split(frame, 2)  # Split frame into two parts

            # Apply DCT to both parts
            X1 = dct(x1, type=2, norm='ortho')
            X2 = dct(x2, type=2, norm='ortho')

            # Define frequency range for modification
            low = int(gamma1 * len(frame))
            high = int(gamma2 * len(frame) + 1)

            # Select the frequency coefficients to modify
            X1p = np.expand_dims(X1[low:high], axis=-1)
            X2p = np.expand_dims(X2[low:high], axis=-1)

            # Perform SVD to decompose the selected coefficients
            u1, s1, v1 = svd(X1p, full_matrices=False)
            u2, s2, v2 = svd(X2p, full_matrices=False)

            l1, l2 = s1[0], s2[0]

            # Modify the singular values based on the watermark bit
            l1p, l2p = modify_svd_pair(self.alpha, l1, l2, message[i])

            s1p = np.array([l1p])
            s2p = np.array([l2p])

            # Reconstruct the modified coefficients
            X1p_em = u1 @ np.diag(s1p) @ v1
            X2p_em = u2 @ np.diag(s2p) @ v2

            # Update the frequency coefficients with the modified values
            X1[low:high] = np.squeeze(X1p_em)
            X2[low:high] = np.squeeze(X2p_em)

            # Apply inverse DCT to reconstruct the modified frames
            x1_em = idct(X1, type=2, norm='ortho')
            x2_em = idct(X2, type=2, norm='ortho')

            # Concatenate the modified frames
            frame_w = np.concatenate([x1_em, x2_em])

            watermarked_frames.append(frame_w)

        # Return the fully watermarked signal
        return np.concatenate(watermarked_frames)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        Extract the watermark message from the watermarked audio signal.

        Args:
            data_with_watermark (np.ndarray): Watermarked audio signal.
            watermark_length (int): Length of the watermark message.

        Returns:
            List[int]: Extracted watermark message bits.
        """
        gamma1 = self.gamma1 * self.sr / 44100
        gamma2 = self.gamma2 * self.sr / 44100

        # Split the watermarked audio into frames based on the watermark length
        frames = np.array_split(data_with_watermark, watermark_length)
        watermark_bits = []

        # Loop through each frame to extract the watermark bits
        for frame in frames:
            x1, x2 = np.array_split(frame, 2)

            # Apply DCT to both parts
            X1 = dct(x1, type=2, norm='ortho')
            X2 = dct(x2, type=2, norm='ortho')

            # Define frequency range for analysis
            low = int(gamma1 * len(frame))
            high = int(gamma2 * len(frame) + 1)

            # Select the frequency coefficients to analyze
            X1p = np.expand_dims(X1[low:high], axis=-1)
            X2p = np.expand_dims(X2[low:high], axis=-1)

            # Perform SVD
            u1, s1, v1 = svd(X1p, full_matrices=False)
            u2, s2, v2 = svd(X2p, full_matrices=False)

            # Extract singular values
            l1, l2 = s1[0], s2[0]

            # Compare the singular values to decode the watermark bit
            watermark_bits.append(0 if l1 / l2 < 1 else 1)

        return watermark_bits

    def type(self) -> str:
        return "Frequency Singular Value Coefficient Modification (FSVC) method"
