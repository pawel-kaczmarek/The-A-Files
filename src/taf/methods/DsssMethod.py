from typing import List
import numpy as np
from taf.models.SteganographyMethod import SteganographyMethod


class DsssMethod(SteganographyMethod):
    """
    Implements the Direct Sequence Spread Spectrum (DSSS) technique for audio watermarking.

    Each bit is spread over one frame by a key-derived bipolar pseudo-noise
    sequence and added at a level proportional to the local frame RMS. The
    receiver correlates every frame against the same sequence and reads the
    bit from the sign of the correlation, which needs neither the cover signal
    nor a known playback gain.
    """

    def __init__(self, key: int = 20240521, alpha: float = 0.05, min_chip_length: int = 1024):
        """
        Args:
            key (int): Seed of the pseudo-noise generator, shared by both sides.
            alpha (float): Embedding strength, as a fraction of the frame RMS.
            min_chip_length (int): Shortest usable frame; sets the capacity.
        """
        self.key = key
        self.alpha = alpha
        self.min_chip_length = min_chip_length

    def _frame_length(self, sample_count: int, bit_count: int) -> int:
        """Frames are sized from the signal so no bit is silently dropped."""
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        frame_length = sample_count // bit_count
        if frame_length < self.min_chip_length:
            capacity = sample_count // self.min_chip_length
            raise ValueError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return frame_length

    def _pseudo_noise(self, frame_length: int) -> np.ndarray:
        rng = np.random.default_rng(self.key)
        return rng.integers(0, 2, frame_length).astype(np.float64) * 2.0 - 1.0

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Encode a watermark message into an audio signal using DSSS.

        Args:
            data (np.ndarray): Input audio signal.
            message (List[int]): Watermark message bits to embed.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        audio = np.asarray(data, dtype=np.float64)
        frame_length = self._frame_length(len(audio), len(message))
        pn = self._pseudo_noise(frame_length)

        stego = audio.copy()
        for index, bit in enumerate(message):
            start = index * frame_length
            frame = stego[start:start + frame_length]

            # Scale the chip to the frame so loud frames hide more energy and
            # quiet frames are not swamped.
            rms = float(np.sqrt(np.mean(np.square(frame))))
            sign = 1.0 if int(bit) == 1 else -1.0
            frame += self.alpha * rms * sign * pn

        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        Decode a watermark message from a watermarked audio signal using DSSS.

        Args:
            data_with_watermark (np.ndarray): Watermarked audio signal.
            watermark_length (int): Length of the watermark message to extract.

        Returns:
            List[int]: Decoded watermark bits.
        """
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        frame_length = self._frame_length(len(audio), watermark_length)
        pn = self._pseudo_noise(frame_length)

        # Speech energy is concentrated at low frequencies while the chip
        # sequence is white, so correlating the raw frame buries the watermark
        # in host interference. A first-difference pre-whitening filter on both
        # sides of the correlation removes most of it.
        whitened_pn = np.diff(pn)

        bits: List[int] = []
        for index in range(watermark_length):
            start = index * frame_length
            frame = audio[start:start + frame_length]
            correlation = float(np.dot(np.diff(frame), whitened_pn))
            bits.append(1 if correlation >= 0 else 0)

        return bits

    def type(self) -> str:
        """Return the type of watermarking method."""
        return "Direct Sequence Spread Spectrum (DSSS) technique"
