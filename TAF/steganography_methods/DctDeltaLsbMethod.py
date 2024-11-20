from typing import List, Tuple

import numpy as np
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography_methods.common.split import to_frames, to_samples


def _normalize(vector: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate the norm and unit vector.

    Args:
        vector (np.ndarray): Input vector.

    Returns:
        Tuple[float, np.ndarray]: Norm and unit vector.
    """
    norm = np.linalg.norm(vector)
    unit_vector = vector / norm if norm != 0 else np.zeros_like(vector)
    return norm, unit_vector


def _denormalize(norm: float, unit_vector: np.ndarray) -> np.ndarray:
    """
    Restore the original vector using the norm and unit vector.

    Args:
        norm (float): Norm value.
        unit_vector (np.ndarray): Unit vector.

    Returns:
        np.ndarray: Restored vector.
    """
    return norm * unit_vector


def _combine_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Combine two vectors into a single array.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.

    Returns:
        np.ndarray: Combined array.
    """
    return np.concatenate((v2, v1))


def _split_vectors(coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the coefficients into two vectors.

    Args:
        coeffs (np.ndarray): DCT coefficients.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two split vectors.
    """
    mid = len(coeffs) // 2
    return coeffs[mid:], coeffs[:mid]


class DctDeltaLsbMethod(SteganographyMethod):
    """
    Implements the DCT Delta LSB method for audio watermarking.

    Attributes:
        sr (int): Sampling rate of the audio signal.
        frame_length_in_ms (int): Frame length in milliseconds for processing.
        delta_value (int): Delta value used for watermark embedding.
    """

    def __init__(self, sr: int, frame_length_in_ms: int = 100, delta_value: int = 10):
        self.sr = sr  # Sampling rate
        self.frame_length_in_ms = frame_length_in_ms  # Frame length in ms
        self.delta_value = delta_value  # Delta for norm adjustment

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Encode the watermark message into the audio signal.

        Args:
            data (np.ndarray): Input audio signal.
            message (List[int]): Watermark bits to embed.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        frames, last_frame = to_frames(data, self.sr, self.frame_length_in_ms)

        for i, bit in enumerate(message):
            coeffs = dct(frames[i], type=2, norm='ortho')
            v1, v2 = _split_vectors(coeffs)

            norm1, u1 = _normalize(v1)
            norm2, u2 = _normalize(v2)
            avg_norm = (norm1 + norm2) / 2

            if bit == 1:
                norm1, norm2 = avg_norm + self.delta_value, avg_norm - self.delta_value
            else:
                norm1, norm2 = avg_norm - self.delta_value, avg_norm + self.delta_value

            v1 = _denormalize(norm1, u1)
            v2 = _denormalize(norm2, u2)

            frames[i] = idct(_combine_vectors(v1, v2), type=2, norm='ortho')

        return to_samples(frames, last_frame)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        Decode the watermark message from the audio signal.

        Args:
            data_with_watermark (np.ndarray): Watermarked audio signal.
            watermark_length (int): Length of the watermark message to extract.

        Returns:
            List[int]: Decoded watermark bits.
        """
        frames, _ = to_frames(data_with_watermark, self.sr, self.frame_length_in_ms)
        decoded_watermark = []

        for i in range(watermark_length):
            coeffs = dct(frames[i], type=2, norm='ortho')
            v1, v2 = _split_vectors(coeffs)

            norm1, _ = _normalize(v1)
            norm2, _ = _normalize(v2)

            bit = 1 if norm1 > norm2 else 0
            decoded_watermark.append(bit)

        return decoded_watermark

    def type(self) -> str:
        """Return the type of watermarking method."""
        return "DCT Delta LSB method"
