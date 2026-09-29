from typing import List
import numpy as np
from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text


class PhaseCodingMethod(SteganographyMethod):

    card = MethodCard(
        title=Text("Phase coding", "Kodowanie fazowe"),
        abbreviation="Phase",
        family="phase",
        purpose="steganography",
        references=(Reference("Alsabhany et al.", 2020, doi="10.1016/j.cosrev.2020.100316"),),
        summary=Text(
            en="Stores the message in Fourier phases of the first audio block.",
            pl="Zapisuje wiadomość w fazach transformaty Fouriera pierwszego bloku dźwięku.",
        ),
        details=Text(
            en=(
                "Sets selected phases to +π/2 or −π/2 and mirrors them to preserve a real waveform. "
                "Subsequent blocks retain their original inter-block phase differences. Decoding reads "
                "the signs of the first block’s phases. Magnitudes are retained during embedding, but a "
                "shift or removal of the opening block can destroy access to the payload."
            ),
            pl=(
                "Ustawia wybrane fazy na +π/2 lub −π/2 i odbija je symetrycznie, aby zachować "
                "rzeczywisty sygnał. Kolejne bloki zachowują pierwotne różnice faz między blokami. "
                "Dekoder odczytuje znaki faz pierwszego bloku. Moduły widma są zachowywane przy "
                "osadzaniu, lecz przesunięcie lub usunięcie początku może uniemożliwić odczyt danych."
            ),
        ),
    )

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        msg_length = len(message)
        chunk_size = int(2 * 2 ** np.ceil(np.log2(2 * msg_length)))
        if chunk_size > data.shape[0]:
            # The whole message sits in the first chunk; a chunk longer than
            # the cover would be cut off when the output is trimmed.
            raise CapacityError(
                f"message too long for cover audio: a {msg_length}-bit message needs a "
                f"{chunk_size}-sample chunk, the cover has {data.shape[0]} samples"
            )
        num_chunks = int(np.ceil(data.shape[0] / chunk_size))
        data_with_watermark = data.copy()

        # Reshaping the data to make it suitable for chunking
        if len(data.shape) == 1:
            data_with_watermark.resize(num_chunks * chunk_size, refcheck=False)
            data_with_watermark = data_with_watermark[np.newaxis]
        else:
            data_with_watermark.resize((num_chunks * chunk_size, data_with_watermark.shape[1]), refcheck=False)
            data_with_watermark = data_with_watermark.T

        # Breaking the data into chunks
        chunks = data_with_watermark[0].reshape((num_chunks, chunk_size))

        # Applying DFT (Discrete Fourier Transform) on audio chunks
        chunks_dft = np.fft.fft(chunks)
        magnitudes = np.abs(chunks_dft)
        phases = np.angle(chunks_dft)
        phase_diff = np.diff(phases, axis=0)

        # Convert the message into binary phase differences
        message_in_pi = np.ravel(message.copy())
        message_in_pi[message_in_pi == 0] = -1  # Convert 0 to -1 for phase representation
        message_in_pi = message_in_pi * -np.pi / 2  # Convert to phase difference in radians

        mid_chunk = chunk_size // 2
        # Embedding the message into the phase of the chunks
        phases[0, mid_chunk - msg_length: mid_chunk] = message_in_pi
        phases[0, mid_chunk + 1: mid_chunk + 1 + msg_length] = -message_in_pi[::-1]

        # Compute the phase matrix for the rest of the chunks
        for i in range(1, len(phases)):
            phases[i] = phases[i - 1] + phase_diff[i - 1]

        # Apply inverse Fourier transform after modifying the phases
        chunks_dft = magnitudes * np.exp(1j * phases)  # Recombine magnitudes and new phases
        chunks = np.fft.ifft(chunks_dft).real  # Perform inverse FFT to return to time domain

        # Reconstruct the watermarked audio data
        data_with_watermark[0] = chunks.ravel().astype(np.float32)

        # Return the watermarked audio with the original length
        return np.squeeze(data_with_watermark.T)[:len(data)]

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        text_length = watermark_length
        block_length = 2 * int(2 ** np.ceil(np.log2(2 * text_length)))
        block_mid = block_length // 2

        # Extract the first block of audio to retrieve the phase information
        if len(data_with_watermark.shape) == 1:
            code = data_with_watermark[:block_length]
        else:
            code = data_with_watermark[:block_length, 0]

        # Get the phase of the DFT of the block
        code_phases = np.angle(np.fft.fft(code))[block_mid - text_length:block_mid]

        # Convert the phase to binary (0 if phase is positive, 1 if negative)
        code_in_binary = (code_phases < 0).astype(np.int16)
        return code_in_binary

    def type(self) -> str:
        return "Phase coding technique"
