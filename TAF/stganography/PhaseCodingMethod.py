from typing import List

import numpy as np

from TAF.models.SteganographyMethod import SteganographyMethod


class PhaseCodingMethod(SteganographyMethod):

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        msgLength = len(message)
        chunkSize = int(2 * 2 ** np.ceil(np.log2(2 * msgLength)))
        numberOfChunks = int(np.ceil(data.shape[0] / chunkSize))
        data_with_watermark = data.copy()

        # Breaking the Audio into chunks
        if len(data.shape) == 1:
            data_with_watermark.resize(numberOfChunks * chunkSize, refcheck=False)
            data_with_watermark = data_with_watermark[np.newaxis]
        else:
            data_with_watermark.resize((numberOfChunks * chunkSize, data_with_watermark.shape[1]), refcheck=False)
            data_with_watermark = data_with_watermark.T

        chunks = data_with_watermark[0].reshape((numberOfChunks, chunkSize))

        # Applying DFT on audio chunks
        chunks = np.fft.fft(chunks)
        magnitudes = np.abs(chunks)
        phases = np.angle(chunks)
        phaseDiff = np.diff(phases, axis=0)

        # Convert message in binary to phase differences
        textInPi = np.ravel(message.copy())
        textInPi[textInPi == 0] = -1
        textInPi = textInPi * -np.pi / 2
        midChunk = chunkSize // 2
        # Phase conversion
        phases[0, midChunk - msgLength: midChunk] = textInPi
        phases[0, midChunk + 1: midChunk + 1 + msgLength] = -textInPi[::-1]
        # Compute the phase matrix
        for i in range(1, len(phases)):
            phases[i] = phases[i - 1] + phaseDiff[i - 1]

        # Apply Inverse fourier transform after applying phase differences
        chunks = (magnitudes * np.exp(1j * phases))
        chunks = np.fft.ifft(chunks).real
        # Combining all block of audio again
        data_with_watermark[0] = chunks.ravel().astype(np.float)

        return np.squeeze(data_with_watermark.T)[:len(data)]

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        textLength = watermark_length
        blockLength = 2 * int(2 ** np.ceil(np.log2(2 * textLength)))
        blockMid = blockLength // 2
        # Get header info
        if len(data_with_watermark.shape) == 1:
            code = data_with_watermark[:blockLength]
        else:
            code = data_with_watermark[:blockLength, 0]
        # Get the phase and convert it to binary
        codePhases = np.angle(np.fft.fft(code))[blockMid - textLength:blockMid]
        codeInBinary = (codePhases < 0).astype(np.int16)
        return codeInBinary

    def type(self) -> str:
        return "Phase coding technique"
