import math
from typing import List

import numpy as np

from TAF.models.SteganographyMethod import SteganographyMethod


def _interpolate_samples(samples: np.ndarray) -> List[int]:
    """Calculate interpolated samples."""
    return [(samples[i] + samples[i + 1]) // 2 for i in range(len(samples) - 1)]


def _least_prime_factor(n: int) -> int:
    """Find the least prime factor of a number."""
    if n < 2:
        return n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


class PrimeFactorInterpolatedMethod(SteganographyMethod):
    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        """Encode the message into the audio samples."""
        interpolated = _interpolate_samples(data)  # Interpolate the audio samples
        message_bits = ''.join(map(str, message))  # Convert the message into a binary string
        stego_samples = []  # List to store stego audio samples
        message_index = 0  # Pointer for the message bits

        # Iterate over the interpolated samples to embed the message
        for i in range(len(interpolated)):
            original = data[i]  # Original sample
            interp = interpolated[i]  # Interpolated sample
            diff = abs(original - interp)  # Calculate the difference between original and interpolated samples

            # Calculate N based on the difference
            N = math.floor(math.log2(diff)) if diff != 0 else 0
            # Ensure sample_space is at least 1
            sample_space = max(_least_prime_factor(N) if N >= 2 else N, 1)

            # Extract a part of the message based on the sample_space
            if message_index < len(message_bits):
                part = int(message_bits[message_index:message_index + sample_space], 2)
                message_index += sample_space
            else:
                part = 0

            # Embed the part of the message into the interpolated sample
            if interp > 0:
                modified_interp = interp + part  # Modify the positive interpolated sample
            elif interp < 0:
                modified_interp = -(abs(interp) + part)  # Modify the negative interpolated sample
            else:
                modified_interp = interp  # If zero, keep it unchanged

            # Append the modified interpolated sample and the original sample
            stego_samples.append(modified_interp)
            stego_samples.append(original)

        # Handle the last sample if the number of samples is odd
        if len(data) % 2 != 0:
            stego_samples.append(data[-1])

        return np.array(stego_samples, dtype=data.dtype)  # Return the stego audio as a numpy array

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        """Decode the message from the stego audio."""
        # Split the samples into original and interpolated parts
        original_samples = data_with_watermark[1::2]  # Odd indexed samples (original)
        interpolated_samples = data_with_watermark[0::2]  # Even indexed samples (interpolated)

        # Variable to store the binary message bits
        message_bits = ''

        # Process each pair of samples
        for i in range(len(interpolated_samples)):
            interp = interpolated_samples[i]  # Interpolated sample
            original = original_samples[i]  # Original sample
            diff = abs(original - interp)  # Calculate the difference between the samples

            # Calculate N based on the difference
            N = math.floor(math.log2(diff)) if diff != 0 else 0
            sample_space = max(_least_prime_factor(N) if N >= 2 else N, 1)

            # If the difference is significant, extract a part of the message
            if diff != 0:
                part = abs(interp - original)  # Usually, we extract the hidden part from this difference
                # Convert the extracted part to binary
                message_bits += format(int(part), f'0{sample_space}b')

        # After collecting all bits, convert them to a list of bits
        # Return only the portion of the message based on the watermark length
        return list(map(int, message_bits[:watermark_length]))

    def type(self) -> str:
        """Return the name of the steganography method."""
        return "Prime Factor Interpolated method"
