from typing import List
import numpy as np
import pywt
from TAF.models.SteganographyMethod import SteganographyMethod


class DwtLsbMethod(SteganographyMethod):
    """
    Implements a Discrete Wavelet Transform (DWT) LSB-based watermarking method.

    This method uses DWT to decompose an audio signal and embeds watermark bits
    into specific coefficients using the Least Significant Bit (LSB) technique.
    """

    def __init__(self, dwt_type: str = 'bior5.5'):
        """
        Initialize the method with the specified DWT wavelet type.

        Args:
            dwt_type (str): Wavelet type for DWT decomposition and reconstruction.
        """
        self.dwt_type = dwt_type

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

        # Embed watermark bits into cD2 coefficients at specific positions
        for i, bit in enumerate(message):
            position = 10 * (i + 1)  # Example position; adjust if needed
            if position < len(cD2):
                cD2[position] = bit
            else:
                break  # Avoid out-of-bounds errors

        # Reconstruct the signal using the modified coefficients
        modified_coeffs = (cA2, cD2, cD1)
        return pywt.waverec(modified_coeffs, self.dwt_type, mode='sym')

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

        # Extract watermark bits from cD2 coefficients at specific positions
        return [
            int(np.rint(cD2[10 * (i + 1)])) if 10 * (i + 1) < len(cD2) else 0
            for i in range(watermark_length)
        ]

    def type(self) -> str:
        """
        Return the type of the watermarking method.

        Returns:
            str: Method description.
        """
        return "DWT LSB-based method"
