from typing import List

import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from TAF.models.SteganographyMethod import SteganographyMethod


def _dct_transform(frame: np.ndarray) -> np.ndarray:
    """Apply DCT to the frame."""
    return dct(frame, norm='ortho')


def _inverse_dct(coeffs: np.ndarray) -> np.ndarray:
    """Apply inverse DCT to the coefficients."""
    return idct(coeffs, norm='ortho')


def _calculate_entropy(sub_band: np.ndarray) -> float:
    """Calculate the entropy of a sub-band."""
    prob = np.histogram(sub_band, bins=256, range=(sub_band.min(), sub_band.max()), density=True)[0]
    prob = prob[prob > 0]  # Avoid log(0)
    return -np.sum(prob * np.log2(prob))


class BlindSvdMethod(SteganographyMethod):

    def __init__(self, frame_size: int = 1024, sub_band_count: int = 4, quantization_coefficient: float = 0.1):
        self.frame_size = frame_size
        self.sub_band_count = sub_band_count
        self.quantization_coefficient = quantization_coefficient

    def _segment_audio(self, audio: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Segment audio signal into non-overlapping frames.
        Returns the segmented frames and the leftover part.
        """
        leftover_size = len(audio) % self.frame_size
        if leftover_size != 0:
            leftover = audio[-leftover_size:]
            audio = audio[:-leftover_size]
        else:
            leftover = np.array([])

        frames = audio.reshape(-1, self.frame_size)
        return frames, leftover

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """Encode the watermark message into the audio signal."""
        frames, leftover = self._segment_audio(data)
        watermarked_frames = []

        for i, frame in enumerate(frames):
            coeffs = _dct_transform(frame)
            low_freq_coeffs = coeffs[:self.frame_size // 2]

            # Divide into sub-bands and calculate entropy
            sub_band_size = len(low_freq_coeffs) // self.sub_band_count
            sub_bands = [low_freq_coeffs[j * sub_band_size:(j + 1) * sub_band_size] for j in range(self.sub_band_count)]
            entropies = [_calculate_entropy(sb) for sb in sub_bands]

            # Select sub-band with maximum entropy
            max_entropy_idx = np.argmax(entropies)
            selected_sub_band = sub_bands[max_entropy_idx]

            # Adjust to nearest perfect square
            original_size = len(selected_sub_band)
            matrix_size = int(np.sqrt(original_size))
            if matrix_size ** 2 > original_size:
                matrix_size -= 1
            adjusted_size = matrix_size ** 2
            selected_matrix = selected_sub_band[:adjusted_size].reshape(matrix_size, matrix_size)

            # Perform SVD
            U, S, Vh = svd(selected_matrix)

            # Quantize and embed watermark into the largest singular value
            Six, Siy = np.cos(np.pi / 4) * S[0], np.sin(np.pi / 4) * S[0]
            Dix = round(Six / self.quantization_coefficient)
            Diy = round(Siy / self.quantization_coefficient)

            # Embed watermark
            watermark_bit = message[i % len(message)]
            Dix_new = Dix + (1 if Dix % 2 != watermark_bit else 0)
            Diy_new = Diy + (1 if Diy % 2 != watermark_bit else 0)

            # Recalculate singular value
            Six_new = Dix_new * self.quantization_coefficient
            Siy_new = Diy_new * self.quantization_coefficient
            S[0] = np.sqrt(Six_new ** 2 + Siy_new ** 2)

            # Reconstruct matrix and flatten back to sub-band
            modified_matrix = U @ np.diag(S) @ Vh
            modified_sub_band = modified_matrix.flatten()

            # Replace back into low-frequency coefficients
            start_idx = max_entropy_idx * sub_band_size
            end_idx = start_idx + len(modified_sub_band)
            low_freq_coeffs[start_idx:end_idx] = modified_sub_band
            coeffs[:self.frame_size // 2] = low_freq_coeffs
            watermarked_frames.append(_inverse_dct(coeffs))

        # Combine watermarked frames and add the leftover part
        watermarked_audio = np.hstack(watermarked_frames)
        if leftover.size > 0:
            watermarked_audio = np.hstack((watermarked_audio, leftover))

        return watermarked_audio

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """Decode the watermark message from the watermarked audio signal."""
        frames, _ = self._segment_audio(data_with_watermark)
        extracted_watermark = []

        for i, frame in enumerate(frames):
            coeffs = _dct_transform(frame)
            low_freq_coeffs = coeffs[:self.frame_size // 2]

            # Divide into sub-bands and calculate entropy
            sub_band_size = len(low_freq_coeffs) // self.sub_band_count
            sub_bands = [low_freq_coeffs[j * sub_band_size:(j + 1) * sub_band_size] for j in range(self.sub_band_count)]
            entropies = [_calculate_entropy(sb) for sb in sub_bands]

            # Select sub-band with maximum entropy
            max_entropy_idx = np.argmax(entropies)
            selected_sub_band = sub_bands[max_entropy_idx]

            # Adjust to nearest perfect square
            original_size = len(selected_sub_band)
            matrix_size = int(np.sqrt(original_size))
            if matrix_size ** 2 > original_size:
                matrix_size -= 1
            adjusted_size = matrix_size ** 2
            selected_matrix = selected_sub_band[:adjusted_size].reshape(matrix_size, matrix_size)

            # Perform SVD
            _, S, _ = svd(selected_matrix)

            # Extract watermark from the largest singular value
            Six, Siy = np.cos(np.pi / 4) * S[0], np.sin(np.pi / 4) * S[0]
            Dix = round(Six / self.quantization_coefficient)
            Diy = round(Siy / self.quantization_coefficient)

            # Decode watermark bit
            extracted_watermark.append(Dix % 2)

            if len(extracted_watermark) >= watermark_length:
                break

        return extracted_watermark

    def type(self) -> str:
        """Return the type of watermarking method."""
        return "Blind SVD-based audio watermarking using entropy and log-polar transformation"
