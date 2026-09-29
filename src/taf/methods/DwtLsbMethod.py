from typing import List
import numpy as np
import pywt
from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text


class DwtLsbMethod(SteganographyMethod):
    """
    Implements a Discrete Wavelet Transform (DWT) LSB-based watermarking method.

    The signal is decomposed with a 2-level DWT and each watermark bit is
    carried by the parity of a quantised detail coefficient, which is the
    wavelet-domain equivalent of flipping a least significant bit.

    The quantisation step is derived from the RMS of the detail band of the
    signal being processed, so the encoder and the decoder agree on it without
    sharing the cover, and the embedding survives amplitude scaling.
    """

    card = MethodCard(
        title="DWT-LSB",
        family="transform",
        purpose="steganography",
        strength_parameter="step_scale",
        references=(Reference("Alsabhany et al.", 2020, doi="10.1016/j.cosrev.2020.100316"),),
        summary=Text(
            en="Hides bits in the parity of quantised wavelet detail coefficients.",
            pl="Ukrywa bity w parzystości skwantowanych współczynników detali falkowych.",
        ),
        details=Text(
            en=(
                "A two-level discrete wavelet transform separates the signal into approximation and "
                "detail bands. Selected detail coefficients are quantised to even or odd indices and "
                "decoded by parity. step_scale controls the step relative to detail-band RMS, making it "
                "follow signal level. Filtering and requantisation can still move coefficients across "
                "decision boundaries."
            ),
            pl=(
                "Dwupoziomowa dyskretna transformacja falkowa rozdziela sygnał na aproksymację i "
                "detale. Wybrane współczynniki detali są kwantowane do indeksów parzystych lub "
                "nieparzystych, z których dekoder odczytuje bity. step_scale określa krok względem RMS "
                "pasma detali, więc skaluje się on z poziomem sygnału. Filtracja i ponowna kwantyzacja "
                "mogą jednak zmienić decyzję dekodera."
            ),
        ),
    )

    def __init__(self, dwt_type: str = 'bior5.5', step_scale: float = 0.5, spacing: int = 10):
        """
        Initialize the method with the specified DWT wavelet type.

        Args:
            dwt_type (str): Wavelet type for DWT decomposition and reconstruction.
            step_scale (float): Quantisation step, as a fraction of the detail
                band RMS. Larger values are more robust and more audible.
            spacing (int): Distance between two consecutive carrier coefficients.
        """
        self.dwt_type = dwt_type
        self.step_scale = step_scale
        self.spacing = spacing

    def _positions(self, coeff_count: int, bit_count: int) -> List[int]:
        positions = [self.spacing * (i + 1) for i in range(bit_count)]
        if positions and positions[-1] >= coeff_count:
            capacity = max(coeff_count // self.spacing - 1, 0)
            raise CapacityError(
                f"message too long for cover audio: {bit_count} > {capacity} bits"
            )
        return positions

    def _step(self, band: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(band))))
        if rms == 0.0:
            raise ValueError("cover audio has an all-zero detail band")
        return self.step_scale * rms

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """
        Embed a watermark message into an audio signal using DWT LSB-based embedding.

        Args:
            data (np.ndarray): Input audio signal.
            message (List[int]): Watermark message bits to embed.

        Returns:
            np.ndarray: Watermarked audio signal.
        """
        # Perform 2-level DWT decomposition
        coeffs = pywt.wavedec(data, self.dwt_type, mode='sym', level=2)
        cA2, cD2, cD1 = coeffs
        cD2 = cD2.copy()

        positions = self._positions(len(cD2), len(message))
        step = self._step(cD2)

        # Carry each bit in the parity of the quantisation index, placing the
        # coefficient at the centre of its quantisation bin.
        for position, bit in zip(positions, message):
            index = int(np.floor(cD2[position] / step))
            if index % 2 != int(bit):
                index += 1
            cD2[position] = (index + 0.5) * step

        # Reconstruct the signal using the modified coefficients
        modified_coeffs = (cA2, cD2, cD1)
        reconstructed = pywt.waverec(modified_coeffs, self.dwt_type, mode='sym')
        return reconstructed[:len(data)]

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """
        Extract a watermark message from a watermarked audio signal.

        Args:
            data_with_watermark (np.ndarray): Watermarked audio signal.
            watermark_length (int): Number of watermark bits to extract.

        Returns:
            List[int]: Extracted watermark bits.
        """
        # Perform 2-level DWT decomposition
        coeffs = pywt.wavedec(data_with_watermark, self.dwt_type, mode='sym', level=2)
        _, cD2, _ = coeffs

        positions = self._positions(len(cD2), watermark_length)
        step = self._step(cD2)

        return [int(np.floor(cD2[position] / step)) % 2 for position in positions]

    def type(self) -> str:
        """
        Return the type of the watermarking method.

        Returns:
            str: Method description.
        """
        return "DWT LSB-based method"
