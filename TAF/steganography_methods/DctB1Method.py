import copy
from typing import List, Tuple

import numpy as np
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod


def _energy_compensation(C: np.ndarray, G2_ind: np.ndarray, niT: float) -> np.ndarray:
    C_hat = C.copy()

    for ind in G2_ind:
        adjustment = max(0, C[ind] ** 2 - niT / len(G2_ind)) ** 0.5
        C_hat[ind] = adjustment if C[ind] >= 0 else -adjustment

    return C_hat


def _smooth_transitions(original_frames: List[np.ndarray], watermarked_frames: List[np.ndarray], lf: int,
                        lt: int) -> List[np.ndarray]:
    for n in range(len(original_frames)):
        alpha = watermarked_frames[n][lt] - original_frames[n][lt]
        beta = watermarked_frames[n][lf - 1] - original_frames[n][lf - 1]

        for k in range(lt):
            transition = alpha * (k + 1) / (lt + 1)
            watermarked_frames[n][k] = original_frames[n][k] + beta + transition

    return watermarked_frames


def _get_energy(C: np.ndarray) -> float:
    return np.sum(np.square(C))


class DctB1Method(SteganographyMethod):
    """
    Implementation of the DCT-b1 audio watermarking method.
    This method embeds watermark bits into the first band of DCT coefficients
    using a masking model and energy compensation.

    Attributes:
        sr (int): Sampling rate of the audio signal.
        lt (int): Number of transition samples in a frame.
        lw (int): Number of samples for embedding in a frame.
        lG1 (int): Number of DCT coefficients in group G1.
        lG2 (int): Number of DCT coefficients in group G2.
    """

    def __init__(self, sr: int, lt: int = 23, lw: int = 1486, lG1: int = 24, lG2: int = 6):
        self.sr = sr  # Sampling rate
        self.lt = lt  # Transition samples
        self.lw = lw  # Embedding samples
        self.lG1 = lG1  # Group G1 size
        self.lG2 = lG2  # Group G2 size
        self.band_size = lG1 + lG2  # Total band size
        self.G1_inds = []

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Encode the watermark message into the audio signal.

        Args:
            data (np.ndarray): Original audio signal.
            message (List[int]): Watermark message bits.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        lf = self.lt + self.lw
        num_frames = len(data) // lf
        bits_per_frame = self.lG1

        # Ensure message length matches the number of frames
        padded_message = (message + [0] * (num_frames * bits_per_frame))[:num_frames * bits_per_frame]

        frames = np.array_split(data[:num_frames * lf], num_frames)
        rframes = []

        for ind, frame in enumerate(frames):
            C = dct(frame[self.lt:], norm="ortho")
            C_hat, G1_ind = self._embed_bits_in_frame(
                C, padded_message[ind * bits_per_frame:(ind + 1) * bits_per_frame]
            )
            self.G1_inds.append(G1_ind)

            # Reconstruct frame
            rframe = np.zeros_like(frame)
            rframe[:self.lt] = frame[:self.lt]
            rframe[self.lt:] = idct(C_hat, norm="ortho")
            rframes.append(rframe)

        # Smooth transitions between frames
        rframes = _smooth_transitions(frames, rframes, lf, self.lt)

        # Combine frames into the final signal
        watermarked_signal = copy.deepcopy(data)
        watermarked_signal[:num_frames * lf] = np.concatenate(rframes)
        return watermarked_signal

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        Decode the watermark message from the watermarked audio signal.

        Args:
            data_with_watermark (np.ndarray): Watermarked audio signal.
            watermark_length (int): Length of the watermark message to decode.

        Returns:
            List[int]: Extracted watermark bits.
        """
        lf = self.lt + self.lw
        num_frames = len(data_with_watermark) // lf
        frames = np.array_split(data_with_watermark[:num_frames * lf], num_frames)
        watermark_bits = []

        for ind, frame in enumerate(frames):
            C = dct(frame[self.lt:], norm="ortho")
            band1 = C[:self.band_size]
            delta = np.sqrt(self._get_band_masking_energy(band1, 0))

            for k in self.G1_inds[ind]:
                bit = 1 if abs(C[k] / delta - np.floor(C[k] / delta) - 0.5) < 0.25 else 0
                watermark_bits.append(bit)

        return watermark_bits[:watermark_length]

    def type(self) -> str:
        """Return the type of watermarking method."""
        return "First band of DCT coefficients (DCT-b1) method"

    def _get_band_representative_frequency(self, band_index: int, num_coeffs: int) -> float:
        start_freq = band_index * self.band_size * self.sr / (2 * num_coeffs)
        end_freq = (band_index + 1) * self.band_size * self.sr / (2 * num_coeffs)
        return (start_freq + end_freq) / 2

    def _get_band_masking_energy(self, C: np.ndarray, band_index: int) -> float:
        freq = self._get_band_representative_frequency(band_index, len(C))
        bark_scale_freq = 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)
        a_tmn = -0.275 * bark_scale_freq - 15.025
        return 10 ** (a_tmn / 10) * _get_energy(C)

    def _divide_band_into_groups(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        choice = np.random.choice(len(C), size=self.lG1, replace=False)
        rest = np.setdiff1d(np.arange(len(C)), choice)
        return np.sort(choice), np.sort(rest)

    def _embed_bits_in_band(self, C: np.ndarray, watermark_bits: List[int], lv: int, band_index: int) -> Tuple[
        np.ndarray, np.ndarray]:
        delta = np.sqrt(self._get_band_masking_energy(C, band_index))
        G1_ind, G2_ind = self._divide_band_into_groups(C)

        C_hat = C.copy()
        for bit, ind in zip(watermark_bits, G1_ind):
            C_hat[ind] = np.floor(C[ind] / delta + 0.5) * delta if bit == 0 else np.floor(
                C[ind] / delta) * delta + delta / 2

        niT = np.sum(C_hat[G1_ind]) - np.sum(C[G1_ind])
        C_hat = _energy_compensation(C_hat, G2_ind, niT)
        return C_hat, G1_ind

    def _embed_bits_in_frame(self, C: np.ndarray, watermark_bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        band1 = C[:self.band_size]
        C_hat, G1_ind1 = self._embed_bits_in_band(band1, watermark_bits, lv=1, band_index=0)
        return C_hat, G1_ind1
