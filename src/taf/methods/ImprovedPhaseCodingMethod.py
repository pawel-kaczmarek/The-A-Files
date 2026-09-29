from typing import List

import numpy as np

from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text


class ImprovedPhaseCodingMethod(SteganographyMethod):

    card = MethodCard(
        title=Text("Improved phase coding", "Ulepszone kodowanie fazowe"),
        abbreviation="IPC",
        family="phase",
        purpose="steganography",
        references=(Reference("Yang", 2024, doi="10.48550/arXiv.2408.13277"),),
        summary=Text(
            en="Distributes phase-coded message portions across multiple Fourier blocks.",
            pl="Rozdziela fragmenty wiadomości kodowanej fazowo pomiędzy wiele bloków Fouriera.",
        ),
        details=Text(
            en=(
                "Divides the payload among segments and sets selected phase pairs to ±π/2 while "
                "retaining spectral magnitudes. The decoder reconstructs the segment layout from the "
                "message length and reads phase signs. Distribution avoids concentrating all bits in "
                "the first segment, but it is not error correction; altered length, phase distortion "
                "and trimming can still corrupt bits."
            ),
            pl=(
                "Dzieli dane między segmenty i ustawia wybrane pary faz na ±π/2, zachowując moduły "
                "widma. Dekoder odtwarza układ segmentów z długości wiadomości i odczytuje znaki faz. "
                "Rozłożenie bitów nie jest kodem korekcyjnym; zmiana długości, zniekształcenia fazowe i "
                "przycinanie nadal mogą uszkodzić dane."
            ),
        ),
    )

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        # Calculate message length in bits
        msg_len = len(message)

        # Calculate segment length, ensuring it's a power of 2
        seg_len = int(2 * 2 ** np.ceil(np.log2(2 * msg_len)))

        # Calculate the number of segments needed
        original_length = len(data)
        if seg_len > original_length:
            # A segment longer than the cover would be mostly padding, and the
            # padding is dropped from the output together with its bits.
            raise CapacityError(
                f"message too long for cover audio: a {msg_len}-bit message needs a "
                f"{seg_len}-sample segment, the cover has {original_length} samples"
            )
        seg_num = int(np.ceil(original_length / seg_len))

        # Zero-pad a copy up to a whole number of segments. Resizing `data`
        # in place would mutate the caller's cover signal.
        data = np.pad(np.asarray(data, dtype=np.float64), (0, seg_num * seg_len - original_length))

        # Convert message to binary representation
        msg_bin = np.ravel(message)

        # Convert binary to phase shifts (-pi/2 for 1, +pi/2 for 0)
        msg_pi = msg_bin.copy()
        msg_pi[msg_pi == 0] = -1
        msg_pi = msg_pi * -np.pi / 2

        # Reshape audio into segments and perform FFT
        segs = data.reshape((seg_num, seg_len))
        segs = np.fft.fft(segs)
        M = np.abs(segs)  # Magnitude
        P = np.angle(segs)  # Phase

        seg_mid = seg_len // 2

        # Embed message into the phase of the middle frequencies
        for i in range(seg_num):
            start = i * len(msg_pi) // seg_num
            end = (i + 1) * len(msg_pi) // seg_num
            P[i, seg_mid - (end - start):seg_mid] = msg_pi[start:end]
            P[i, seg_mid + 1:seg_mid + 1 + (end - start)] = -msg_pi[start:end][::-1]

        # Reconstruct the audio with modified phase and drop the padding so the
        # stego signal keeps the cover length.
        segs = M * np.exp(1j * P)
        return np.fft.ifft(segs).real.ravel()[:original_length].astype(np.float32)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        seg_len = int(2 * 2 ** np.ceil(np.log2(2 * watermark_length)))
        seg_num = int(np.ceil(len(data_with_watermark) / seg_len))
        seg_mid = seg_len // 2

        # Mirror the zero padding the encoder used so the final (partial)
        # segment is still a full-length FFT block.
        data_with_watermark = np.pad(
            np.asarray(data_with_watermark, dtype=np.float64),
            (0, seg_num * seg_len - len(data_with_watermark)),
        )

        extracted_bits = []

        # Extract the embedded message from the phase of the middle frequencies
        for i in range(seg_num):
            x = np.fft.fft(data_with_watermark[i * seg_len:(i + 1) * seg_len])
            extracted_phase = np.angle(x)
            start = i * watermark_length // seg_num
            end = (i + 1) * watermark_length // seg_num
            extracted_bits.extend((extracted_phase[seg_mid - (end - start):seg_mid] < 0).astype(np.int8))

        return np.array(extracted_bits[:watermark_length]).tolist()

    def type(self) -> str:
        return "Improved phase coding technique"
