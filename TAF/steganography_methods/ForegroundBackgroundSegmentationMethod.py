from typing import List

import bitstring
import numpy as np

from TAF.models.SteganographyMethod import SteganographyMethod
from TAF.steganography_methods.common.background_separator import separate_fg_bg_full


class ForegroundBackgroundSegmentationMethod(SteganographyMethod):

    def __init__(self, sr: int, seed: int = 42):
        self.sr = sr
        self.seed = seed

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        data = data.copy()
        fg_mask = separate_fg_bg_full(data, self.sr)

        foreground_indices = np.where(fg_mask)[0]
        background_indices = np.where(~fg_mask)[0]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(foreground_indices)
        rng.shuffle(background_indices)

        msg_idx = 0

        # foreground → 2 bits
        for idx in foreground_indices:
            if msg_idx >= len(message):
                break

            bits = message[msg_idx:msg_idx + 2]
            bit_array = bitstring.BitArray(float=data[idx], length=32).bin

            if len(bits) == 2:
                bit_array = bit_array[:-2] + ''.join(map(str, bits))
                msg_idx += 2
            else:
                bit_array = bit_array[:-1] + str(bits[0])
                msg_idx += 1

            data[idx] = bitstring.BitArray(bin=bit_array, length=32).float

        # background → 1 bit
        if msg_idx < len(message):
            for idx in background_indices:
                if msg_idx >= len(message):
                    break

                bit_array = bitstring.BitArray(float=data[idx], length=32).bin
                bit_array = bit_array[:-1] + str(message[msg_idx])
                data[idx] = bitstring.BitArray(bin=bit_array, length=32).float
                msg_idx += 1

        return data

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:

        fg_mask = separate_fg_bg_full(data_with_watermark, self.sr)

        foreground_indices = np.where(fg_mask)[0]
        background_indices = np.where(~fg_mask)[0]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(foreground_indices)
        rng.shuffle(background_indices)

        extracted = []

        for idx in foreground_indices:
            if len(extracted) >= watermark_length:
                break

            bit_array = bitstring.BitArray(float=data_with_watermark[idx], length=32).bin
            extracted.extend([int(bit_array[-2]), int(bit_array[-1])])

        if len(extracted) < watermark_length:
            for idx in background_indices:
                if len(extracted) >= watermark_length:
                    break

                bit_array = bitstring.BitArray(float=data_with_watermark[idx], length=32).bin
                extracted.append(int(bit_array[-1]))

        return extracted[:watermark_length]

    def type(self) -> str:
        return "Foreground-Background Segmentation LSB (FBS-LSB)"