"""Echo hiding with paired backward and forward kernels.

Reference:
    Kim, H. J., & Choi, Y. H. (2003). "A novel echo-hiding scheme with
    backward and forward kernels." IEEE Transactions on Circuits and Systems
    for Video Technology, 13(8), 885-889.
    https://doi.org/10.1109/TCSVT.2003.815950

A single-echo kernel, as used by the repository's EchoMethod, trades audio
quality against detection directly: the echo has to be loud enough for its
cepstrum peak to stand out. Pairing a backward echo with a forward one of
opposite sign puts two peaks in the cepstrum for the same added energy, and
the detector differences them, so half the amplitude gives the same margin.
"""
from typing import List

import numpy as np
from scipy.fft import fft, ifft

from taf.models.SteganographyMethod import SteganographyMethod


class BackwardForwardEchoMethod(SteganographyMethod):
    """Echo hiding using a backward/forward kernel pair."""

    def __init__(self, alpha: float = 0.1, d0: int = 150, d1: int = 200,
                 min_frame_length: int = 2048):
        """
        Args:
            alpha: Echo amplitude. Half the amplitude of the single-echo
                scheme gives a comparable detection margin.
            d0: Echo delay carrying a 0 bit, in samples.
            d1: Echo delay carrying a 1 bit, in samples.
            min_frame_length: Shortest usable frame; sets the capacity.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if min(d0, d1) < 1:
            raise ValueError("echo delays must be positive")

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
            raise ValueError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return frame_length

    def _echo(self, frame: np.ndarray, delay: int) -> np.ndarray:
        """Backward plus forward echo at the same delay.

        The two echoes share a sign: the kernel is then 1 + 2*alpha*cos(w*d),
        whose log-magnitude has its cepstral peak at the delay itself. Giving
        them opposite signs instead makes the kernel purely imaginary, and its
        log-magnitude peaks at twice the delay, where no detector looks.
        """
        backward = np.zeros_like(frame)
        backward[delay:] = frame[:-delay]

        forward = np.zeros_like(frame)
        forward[:-delay] = frame[delay:]

        return self.alpha * (backward + forward)

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        frame_length = self._frame_length(len(audio), len(message))
        stego = audio.copy()

        for index, bit in enumerate(message):
            start = index * frame_length
            frame = audio[start:start + frame_length]
            delay = self.d1 if int(bit) == 1 else self.d0
            stego[start:start + frame_length] = frame + self._echo(frame, delay)

        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        frame_length = self._frame_length(len(audio), watermark_length)

        bits: List[int] = []
        for index in range(watermark_length):
            start = index * frame_length
            frame = audio[start:start + frame_length]

            spectrum = np.abs(fft(frame))
            cepstrum = np.real(ifft(np.log(spectrum + 1e-10)))

            # The kernel is symmetric in time, so the cepstrum carries the
            # same peak at +d and -d; summing them doubles the margin for the
            # same added energy.
            score_zero = cepstrum[self.d0] + cepstrum[-self.d0]
            score_one = cepstrum[self.d1] + cepstrum[-self.d1]
            bits.append(1 if score_one >= score_zero else 0)

        return bits

    def type(self) -> str:
        return "Echo hiding with backward and forward kernels"
