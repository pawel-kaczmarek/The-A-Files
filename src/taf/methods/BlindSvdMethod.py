from typing import List

import numpy as np
from scipy.fftpack import dct, idct
from scipy.linalg import svd

from taf.models.SteganographyMethod import SteganographyMethod


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
        """
        Args:
            frame_size: Samples per frame; one bit is carried per frame.
            sub_band_count: Number of low-frequency sub-bands to choose from.
            quantization_coefficient: Quantisation step, relative to the
                energy of the singular values the embedding leaves alone. A
                relative step keeps the method usable after a volume change,
                which a fixed step does not survive.
        """
        self.frame_size = frame_size
        self.sub_band_count = sub_band_count
        self.quantization_coefficient = quantization_coefficient

    def _quantization_step(self, singular_values: np.ndarray) -> float:
        """Derive the step from the singular values that are never modified.

        Only S[0] carries the watermark, so the energy of S[1:] is identical
        on both sides and scales with the signal level.
        """
        residual_energy = float(np.sqrt(np.sum(np.square(singular_values[1:]))))
        if residual_energy == 0.0:
            return self.quantization_coefficient
        return self.quantization_coefficient * residual_energy

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

    def _check_capacity(self, frame_count: int, watermark_length: int, name: str) -> None:
        """One bit per frame; the surplus used to be dropped without warning."""
        if watermark_length > frame_count:
            raise ValueError(
                f"{name} too long for cover audio: {watermark_length} > {frame_count} bits "
                f"({self.frame_size}-sample frames)"
            )

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """Encode the watermark message into the audio signal."""
        frames, leftover = self._segment_audio(data)
        self._check_capacity(len(frames), len(message), "message")
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
            step = self._quantization_step(S)
            Six, Siy = np.cos(np.pi / 4) * S[0], np.sin(np.pi / 4) * S[0]
            Dix = round(Six / step)
            Diy = round(Siy / step)

            # Embed watermark
            watermark_bit = message[i % len(message)]
            Dix_new = Dix + (1 if Dix % 2 != watermark_bit else 0)
            Diy_new = Diy + (1 if Diy % 2 != watermark_bit else 0)

            # Recalculate singular value
            Six_new = Dix_new * step
            Siy_new = Diy_new * step
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
        self._check_capacity(len(frames), watermark_length, "watermark")
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
            step = self._quantization_step(S)
            Six, Siy = np.cos(np.pi / 4) * S[0], np.sin(np.pi / 4) * S[0]
            Dix = round(Six / step)
            Diy = round(Siy / step)

            # Decode watermark bit
            extracted_watermark.append(Dix % 2)

            if len(extracted_watermark) >= watermark_length:
                break

        return extracted_watermark

    def type(self) -> str:
        """Return the type of watermarking method."""
        return "Blind SVD-based audio watermarking using entropy-selected sub-bands"
