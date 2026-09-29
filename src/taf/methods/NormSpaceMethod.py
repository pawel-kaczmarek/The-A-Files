from typing import List
import numpy as np
import pywt
from scipy.fft import dct, idct

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.methods.common.split import to_frames
from taf.models.card import MethodCard, Reference, Text


class NormSpaceMethod(SteganographyMethod):

    card = MethodCard(
        title="Norm-space",
        family="transform",
        purpose="watermarking",
        strength_parameter="delta",
        references=(Reference("Saadi et al.", 2019, doi="10.1016/j.sigpro.2018.08.011"),),
        summary=Text(
            en="Represents each bit by which of two transform-domain vectors has the larger norm.",
            pl=(
                "Reprezentuje bit przez wskazanie, który z dwóch wektorów w dziedzinie transformacji ma "
                "większą normę."
            ),
        ),
        details=Text(
            en=(
                "Applies a Haar DWT and then DCT to the approximation band. Even and odd DCT "
                "coefficients form two vectors whose norms are separated by a relative delta. The "
                "decoder compares them without the original recording. Increasing delta strengthens the "
                "separation but changes audio more; cropping can break the segment grid."
            ),
            pl=(
                "Wykonuje DWT Haara, a następnie DCT pasma aproksymacji. Parzyste i nieparzyste "
                "współczynniki tworzą wektory, których normy rozsuwa względny parametr delta. Dekoder "
                "porównuje je bez oryginalnego nagrania. Większa delta wzmacnia różnicę kosztem "
                "większych zmian dźwięku; przycięcie może zaburzyć podział na segmenty."
            ),
        ),
    )

    #: Each bit needs a segment whose DWT approximation, split into even and
    #: odd DCT coefficients, leaves both sub-vectors non-empty.
    MIN_SAMPLES_PER_BIT = 4

    def __init__(self, sr: int, delta: float = 0.05):
        """
        Args:
            sr: Sampling rate of the audio signal.
            delta: Norm split, as a fraction of the mean sub-vector norm. A
                relative value keeps both the audible distortion and the
                detection margin proportional to the segment level; the
                absolute delta used previously destroyed quiet segments.
        """
        self.sr = sr
        self.delta = delta

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        if len(message) * self.MIN_SAMPLES_PER_BIT > len(data):
            raise CapacityError(
                f"message too long for cover audio: {len(message)} > "
                f"{len(data) // self.MIN_SAMPLES_PER_BIT} bits"
            )
        segments, last_frame = to_frames(data, self.sr, len(data) / self.sr / len(message) * 1000)
        segments = segments.copy()
        rsegments = []

        for ind, segment in enumerate(segments):
            if ind >= len(message):
                # Framing can yield more segments than bits; leave the surplus
                # untouched rather than indexing past the message.
                rsegments.append(segment)
                continue

            cA1, cD1 = pywt.dwt(segment, 'db1')

            v = dct(cA1, norm='ortho')

            v1 = v[::2]
            v2 = v[1::2]

            nrmv1 = np.linalg.norm(v1, ord=2)
            nrmv2 = np.linalg.norm(v2, ord=2)

            u1 = v1 / nrmv1
            u2 = v2 / nrmv2

            watermark_bit = message[ind]
            nrm = (nrmv1 + nrmv2) / 2
            delta = self.delta * nrm
            if watermark_bit == 1:
                nrmv1 = nrm + delta
                nrmv2 = nrm - delta
            else:
                nrmv1 = nrm - delta
                nrmv2 = nrm + delta

            rv1 = nrmv1 * u1
            rv2 = nrmv2 * u2

            rv = np.zeros((len(v),))

            rv[::2] = rv1
            rv[1::2] = rv2

            rcA1 = idct(rv, norm='ortho')

            # idwt pads odd-length segments back up, which previously grew the
            # stego signal by one sample per frame and desynchronised it.
            rseg = pywt.idwt(rcA1, cD1, 'db1')
            rsegments.append(rseg[:len(segment)])

        if last_frame is not None:
            rsegments.append(last_frame)
        return np.concatenate(rsegments)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        segments, last_frame = to_frames(data_with_watermark, self.sr,
                                         len(data_with_watermark) / self.sr / watermark_length * 1000)
        segments = segments.copy()
        watermark_bits = []

        for ind, segment in enumerate(segments):
            cA1, cD1 = pywt.dwt(segment, 'db1')

            v = dct(cA1, norm='ortho')

            v1 = v[::2]
            v2 = v[1::2]

            nrmv1 = np.linalg.norm(v1, ord=2)
            nrmv2 = np.linalg.norm(v2, ord=2)

            if nrmv1 > nrmv2:
                watermark_bits.append(1)
            else:
                watermark_bits.append(0)

            if len(watermark_bits) == watermark_length:
                break

        return watermark_bits

    def type(self) -> str:
        return "Norm space method"
