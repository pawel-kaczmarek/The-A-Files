from typing import List
import bitstring
import numpy as np
from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.card import MethodCard, Reference, Text


class LsbMethod(SteganographyMethod):

    card = MethodCard(
        title="LSB",
        family="lsb",
        purpose="steganography",
        references=(Reference("Alsabhany et al.", 2020, doi="10.1016/j.cosrev.2020.100316"),),
        summary=Text(
            en="Hides one message bit in each audio sample with a very small numerical change.",
            pl=(
                "Ukrywa jeden bit wiadomości w każdej próbce dźwięku, bardzo nieznacznie zmieniając jej "
                "wartość."
            ),
        ),
        details=Text(
            en=(
                "Replaces the least significant bit of the sample’s 32-bit floating-point "
                "representation. Decoding reads those bits in order. This implementation uses float "
                "bits, not integer PCM bits: conversion to PCM, lossy compression or even small signal "
                "processing changes can erase the message."
            ),
            pl=(
                "Zastępuje najmniej znaczący bit 32-bitowej reprezentacji zmiennoprzecinkowej próbki. "
                "Dekoder odczytuje te bity po kolei. Ta implementacja operuje na bitach float, a nie "
                "całkowitoliczbowego PCM: konwersja do PCM, kompresja stratna lub drobne przetwarzanie "
                "sygnału mogą usunąć wiadomość."
            ),
        ),
    )

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        if len(message) > len(data):
            raise CapacityError(
                f"message too long for cover audio: {len(message)} > {len(data)} bits"
            )

        # Work on a copy: the caller keeps the cover to compute quality metrics
        # against, and embedding in place would overwrite it.
        data = data.copy()
        for idx, m in enumerate(message):
            # Convert the floating-point number to a 32-bit binary string
            bit_array = bitstring.BitArray(float=data[idx], length=32).bin
            # Replace the least significant bit (LSB) with the message bit
            bit_array = bit_array[:-1] + str(m)
            # Convert the modified binary string back to a float and update the data array
            data[idx] = bitstring.BitArray(bin=bit_array, length=32).float
        return data

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        # Extract the LSB from each sample and reconstruct the message
        return [
            int(bitstring.BitArray(float=data_with_watermark[x], length=32).bin[-1]) for x in range(watermark_length)
        ]

    def type(self) -> str:
        return "Standard LSB coding method"
