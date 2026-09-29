"""Time-spread echo hiding.

Reference:
    Ko, B.-S., Nishimura, R., & Suzuki, Y. (2005). "Time-spread echo method
    for digital audio watermarking." IEEE Transactions on Multimedia, 7(2),
    212-221. https://doi.org/10.1109/tmm.2005.843366

A single echo is a discrete repetition and is audible as one once it is loud
enough to detect. Spreading the same energy over a key-derived pseudo-noise
sequence turns the echo into a low-level noise-like reverberation, and the
receiver recovers the detection peak by correlating the cepstrum with that
sequence, which concentrates the spread energy back into one point.

The bit is carried by which of two delays the correlation peaks at, so, like
the other echo methods here, extraction needs neither the cover nor a known
playback gain.
"""
from typing import List

import numpy as np
from scipy.fft import fft, ifft

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text


class TimeSpreadEchoMethod(SteganographyMethod):
    """Echo hiding with a pseudo-noise-spread kernel."""

    card = MethodCard(
        title=Text("Time-spread echo", "Echo rozproszone w czasie"),
        abbreviation="TS-Echo",
        family="echo",
        purpose="watermarking",
        strength_parameter="alpha",
        references=(Reference("Ko et al.", 2005, doi="10.1109/TMM.2005.843366"),),
        summary=Text(
            en="Distributes the echo watermark over multiple delays using a keyed sequence.",
            pl="Rozkłada echo znaku wodnego na wiele opóźnień za pomocą sekwencji zależnej od klucza.",
        ),
        details=Text(
            en=(
                "A pseudo-noise kernel spreads echo energy in time instead of concentrating it at one "
                "delay. The decoder uses the corresponding sequence to detect the bit from the cepstral "
                "response. alpha sets strength and the shared seed reproduces the kernel. Spreading "
                "changes the audibility and detection trade-off; it still needs enough audio per bit "
                "and correct alignment."
            ),
            pl=(
                "Jądro pseudolosowe rozprasza energię echa w czasie zamiast skupiać ją w jednym "
                "opóźnieniu. Dekoder używa odpowiadającej sekwencji do odczytania bitu z odpowiedzi "
                "cepstralnej. alpha określa siłę, a wspólne ziarno odtwarza jądro. Rozpraszanie zmienia "
                "kompromis między słyszalnością a detekcją; nadal potrzeba odpowiedniej ilości dźwięku "
                "na bit i synchronizacji."
            ),
        ),
    )

    def __init__(
        self,
        key: int = 20240521,
        alpha: float = 0.1,
        pn_length: int = 511,
        d0: int = 200,
        d1: int = 900,
        min_frame_length: int = 4096,
    ):
        """
        Args:
            key: Seed of the pseudo-noise sequence, shared by both sides.
            alpha: Total echo amplitude. The sequence is normalised to unit
                norm, so alpha is the energy of the whole spread echo rather
                than of each individual chip.
            pn_length: Length of the spreading sequence, in samples.
            d0: Kernel offset carrying a 0 bit, in samples.
            d1: Kernel offset carrying a 1 bit, in samples.
            min_frame_length: Shortest usable frame; sets the capacity.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if pn_length < 2:
            raise ValueError("pn_length must be at least 2")
        if min(d0, d1) < 1:
            raise ValueError("kernel offsets must be positive")

        self.key = key
        self.alpha = alpha
        self.pn_length = pn_length
        self.d0 = d0
        self.d1 = d1
        self.min_frame_length = min_frame_length

    def _frame_length(self, sample_count: int, bit_count: int) -> int:
        if bit_count <= 0:
            raise ValueError("message must not be empty")

        frame_length = sample_count // bit_count
        needed = max(self.min_frame_length, self.d0 + self.pn_length, self.d1 + self.pn_length)
        if frame_length < needed:
            capacity = sample_count // needed
            raise CapacityError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return frame_length

    def _pseudo_noise(self) -> np.ndarray:
        rng = np.random.default_rng(self.key)
        chips = rng.integers(0, 2, self.pn_length).astype(np.float64) * 2.0 - 1.0
        # Unit norm: without it the echo energy grows with the sequence
        # length, and a 511-chip kernel at alpha=0.05 buries the signal.
        return chips / np.sqrt(self.pn_length)

    def _kernel(self, frame_length: int, offset: int, pn: np.ndarray) -> np.ndarray:
        """Unit impulse followed by the spread echo at the given offset."""
        kernel = np.zeros(frame_length)
        kernel[0] = 1.0
        kernel[offset:offset + len(pn)] = self.alpha * pn
        return kernel

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        audio = np.asarray(data, dtype=np.float64)
        frame_length = self._frame_length(len(audio), len(message))
        pn = self._pseudo_noise()
        stego = audio.copy()

        for index, bit in enumerate(message):
            start = index * frame_length
            frame = audio[start:start + frame_length]
            offset = self.d1 if int(bit) == 1 else self.d0

            # Circular convolution keeps the frame length exactly, and the
            # cepstral detector is itself circular.
            spectrum = fft(frame) * fft(self._kernel(frame_length, offset, pn))
            stego[start:start + frame_length] = np.real(ifft(spectrum))

        return stego.astype(data.dtype, copy=False)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        audio = np.asarray(data_with_watermark, dtype=np.float64)
        frame_length = self._frame_length(len(audio), watermark_length)
        pn = self._pseudo_noise()

        bits: List[int] = []
        for index in range(watermark_length):
            start = index * frame_length
            frame = audio[start:start + frame_length]

            cepstrum = np.real(ifft(np.log(np.abs(fft(frame)) + 1e-10)))

            # Correlating the cepstrum with the spreading sequence gathers the
            # spread echo back into a single peak at its offset.
            score_zero = float(np.dot(cepstrum[self.d0:self.d0 + len(pn)], pn))
            score_one = float(np.dot(cepstrum[self.d1:self.d1 + len(pn)], pn))
            bits.append(1 if score_one >= score_zero else 0)

        return bits

    def type(self) -> str:
        return "Time-spread echo method"
