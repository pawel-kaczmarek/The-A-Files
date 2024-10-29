from typing import List

import numpy as np

from TAF.models.SteganographyMethod import SteganographyMethod


class ImprovedPhaseCodingMethod(SteganographyMethod):

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        # Calculate message length in bits
        msg_len = len(message)
        # Calculate segment length, ensuring it's a power of 2
        seg_len = int(2 * 2 ** np.ceil(np.log2(2 * msg_len)))
        # Calculate the number of segments needed
        seg_num = int(np.ceil(len(data) / seg_len))

        # Resize the audio array to fit the number of segments
        data.resize(seg_num * seg_len, refcheck=False)

        # Convert message to binary representation
        msg_bin = np.ravel(message)
        # Convert binary to phase shifts (-pi/8 for 1, pi/8 for 0)
        msg_pi = msg_bin.copy()
        msg_pi[msg_pi == 0] = -1
        msg_pi = msg_pi * -np.pi / 2  # Use smaller phase to improve audio quality 1/8 may cause low BER, so change back to 1/2

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

        # Reconstruct the audio with modified phase
        segs = M * np.exp(1j * P)
        return np.fft.ifft(segs).real.ravel().astype(np.float32)

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        seg_len = int(2 * 2 ** np.ceil(np.log2(2 * watermark_length)))
        seg_num = int(np.ceil(len(data_with_watermark) / seg_len))
        seg_mid = seg_len // 2

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
