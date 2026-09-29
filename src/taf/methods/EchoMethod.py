from typing import List
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import lfilter
from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.methods.common.mixer import mixer
from taf.models.card import MethodCard, Reference, Text


class EchoMethod(SteganographyMethod):
    """
    Implements the Echo Hiding watermarking technique using single echo kernels.
    This method embeds watermark bits by applying echoes with different delays to an audio signal.

    Frames are sized from the signal so that every message bit gets one, and
    the echo amplitude is a constructor parameter: the original fixed value of
    0.5 produced a plainly audible echo (SNR 6 dB on speech).
    """

    card = MethodCard(
        title=Text("Echo hiding", "Ukrywanie w echu"),
        abbreviation="Echo",
        family="echo",
        purpose="steganography",
        strength_parameter="alpha",
        needs_long_input=True,
        references=(Reference("Alsabhany et al.", 2020, doi="10.1016/j.cosrev.2020.100316"),),
        summary=Text(
            en="Represents zero and one by echoes with different delays.",
            pl="Reprezentuje zero i jedynkę przez echa o różnych opóźnieniach.",
        ),
        details=Text(
            en=(
                "Splits the recording into payload frames and mixes in a delayed copy using alpha. The "
                "decoder detects the selected delay through cepstral analysis. Echo strength trades "
                "audibility against detection margin. Short frames, existing reverberation or temporal "
                "changes can make the two delay hypotheses difficult to distinguish."
            ),
            pl=(
                "Dzieli nagranie na ramki wiadomości i domiesza opóźnioną kopię z siłą alpha. Dekoder "
                "wykrywa wybrane opóźnienie przez analizę cepstralną. Siła echa wiąże słyszalność z "
                "marginesem detekcji. Krótkie ramki, istniejący pogłos lub zmiany czasowe mogą utrudnić "
                "odróżnienie obu opóźnień."
            ),
        ),
    )

    def __init__(self, alpha: float = 0.2, d0: int = 150, d1: int = 200, min_frame_length: int = 2048):
        """
        Args:
            alpha (float): Echo amplitude. Higher is more robust, more audible.
            d0 (int): Echo delay carrying a 0 bit, in samples.
            d1 (int): Echo delay carrying a 1 bit, in samples.
            min_frame_length (int): Shortest usable frame; sets the capacity.
        """
        self.alpha = alpha
        self.d0 = d0
        self.d1 = d1
        self.min_frame_length = min_frame_length

    def _frame_length(self, sample_count: int, bit_count: int) -> int:
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        frame_length = sample_count // bit_count
        if frame_length < self.min_frame_length:
            capacity = sample_count // self.min_frame_length
            raise CapacityError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return frame_length

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Embed a watermark message into an audio signal using echo hiding.

        Args:
            data (np.ndarray): Input audio signal.
            message (List[int]): Watermark message bits to embed.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        d0, d1, alpha = self.d0, self.d1, self.alpha
        L = self._frame_length(len(data), len(message))
        N = len(message)
        bits = list(message)

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
        d0, d1 = self.d0, self.d1
        L = self._frame_length(len(data_with_watermark), watermark_length)
        N = watermark_length
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
