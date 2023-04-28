import copy
from typing import List

import numpy as np
from scipy.fft import dct, idct

from TAF.models.SteganographyMethod import SteganographyMethod


class DctB1Method(SteganographyMethod):

    def __init__(self,
                 sr: int,
                 lt: int = 23,
                 lw: int = 1486,
                 lG1: int = 24,
                 lG2: int = 6):
        self.sr = sr  # sampling rate
        self.lt = lt  # number of transition samples in a frame
        self.lw = lw  # number of samples for embedding in a frame
        self.lG1 = lG1  # number of DCT coefficients in group G1
        self.lG2 = lG2  # number of DCT coefficients in group G2
        self.band_size = lG1 + lG2  # number of DCT coefficients in a band. It must be equal to lG1+lG2

    def encode(self, data: np.ndarray, message: List[int]) -> np.ndarray:
        lf = self.lt + self.lw
        signal_length = len(data)
        num_frames = signal_length // lf

        bits_per_frame = self.lG1
        if len(message) != num_frames * bits_per_frame:
            message = (message + num_frames * bits_per_frame * [0])[:num_frames * bits_per_frame]

        frames = np.array_split(data[:(num_frames * lf)], num_frames)
        rframes = []

        G1_inds = []

        watermarked_signal = copy.deepcopy(data)
        for ind, frame in enumerate(frames):
            C = dct(frame[self.lt:], norm="ortho")
            C_hat, G1_ind = self.embed_bits_in_frame(C,
                                                     message[(ind * bits_per_frame):((ind + 1) * bits_per_frame)])
            G1_inds.append(G1_ind)

            rframe = np.zeros(frame.shape)
            rframe[:self.lt] = frame[:self.lt]
            rframe[self.lt:] = idct(C_hat, norm="ortho")
            rframes.append(rframe)

        rframes = self.smooth_transitions(frames, rframes, lf, self.lt)
        watermarked_signal[:(num_frames * lf)] = np.concatenate(rframes)
        self.G1_inds = G1_inds
        return watermarked_signal

    def decode(self, data_with_watermark: np.ndarray, watermark_length: int) -> List[int]:
        lf = self.lt + self.lw
        signal_length = len(data_with_watermark)
        num_frames = signal_length // lf
        frames = np.array_split(data_with_watermark[:(num_frames * lf)], num_frames)
        watermark_bits = []

        for ind, frame in enumerate(frames):
            C = dct(frame[self.lt:], norm="ortho")
            band1 = C[:self.band_size]

            delta = np.sqrt(self.get_band_masking_energy(band1, 0))

            for k in self.G1_inds[ind]:
                if abs(C[k] / delta - np.floor(C[k] / delta) - 0.5) < 0.25:
                    watermark_bits.append(1)
                else:
                    watermark_bits.append(0)

        return watermark_bits[:watermark_length]

    def type(self) -> str:
        return "First band of DCT coefficients (DCT-b1)  method"

    def get_band_representative_frequency(self, band_index, num_coeffs):
        start_freq = band_index * self.band_size * self.sr / (2 * num_coeffs)
        end_freq = (band_index + 1) * self.band_size * self.sr / (2 * num_coeffs)

        return (start_freq + end_freq) / 2

    def get_band_masking_energy(self, C, band_index):
        freq = self.get_band_representative_frequency(band_index, len(C))

        bark_scale_freq = 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)

        a_tmn = -0.275 * bark_scale_freq - 15.025

        return 10 ** (a_tmn / 10) * self.get_energy(C)

    def divide_band_into_groups(self, C):
        choice = np.random.choice(range(C.shape[0]), size=(self.lG1,), replace=False)
        rest = np.array([i for i in range(C.shape[0]) if i not in choice])

        return np.sort(choice), np.sort(rest)

    def energy_compensation(self, C, G2_ind, niT):
        C_hat = C[:]

        if niT < 0:
            for ind in G2_ind:
                if C[ind] >= 0:
                    C_hat[ind] = (C[ind] ** 2 - niT / self.lG2) ** (1 / 2)
                else:
                    C_hat[ind] = -(C[ind] ** 2 - niT / self.lG2) ** (1 / 2)

        elif niT > 0:
            ni = niT
            G2_ind_sorted = sorted(list(G2_ind), key=lambda x: abs(C[x]))
            for k, ind in enumerate(G2_ind_sorted):
                if C[ind] >= 0:
                    C_hat[ind] = (max(0, C[ind] ** 2 - ni / (lG2 - (k + 1)))) ** (1 / 2)
                else:
                    C_hat[ind] = -(max(0, C[ind] ** 2 - ni / (lG2 - (k + 1)))) ** (1 / 2)

                ni = ni - (C[ind] ** 2 - C_hat[ind] ** 2)

        return C_hat

    def embed_bits_in_band(self, C, watermark_bits, lv, band_index):
        delta = np.sqrt(self.get_band_masking_energy(C, band_index))
        delta_sigma = np.sqrt(lv) * delta

        G1_ind, G2_ind = self.divide_band_into_groups(C)
        C_hat = C[:]
        for i, ind in enumerate(G1_ind):
            wb = watermark_bits[i]
            if wb == 0:
                C_hat[ind] = np.floor(C[ind] / delta + 0.5) * delta
            else:
                C_hat[ind] = np.floor(C[ind] / delta) * delta + delta / 2

        niT = np.sum([C_hat[k] for k in G1_ind]) - np.sum([C[k] for k in G1_ind])

        C_hat = self.energy_compensation(C, G2_ind, niT)
        return C_hat, G1_ind

    def embed_bits_in_frame(self, C, watermark_bits):
        band1 = C[:self.band_size]
        C_hat = C[:]
        C_hat[:self.band_size], G1_ind1 = self.embed_bits_in_band(band1, watermark_bits, 1, 0)

        return C_hat, G1_ind1

    @staticmethod
    def smooth_transitions(original_frames, watermarked_frames, lf, lt):
        alphas = np.zeros(len(original_frames))
        bethas = np.zeros(len(original_frames))

        for n in range(len(original_frames)):
            alphas[n] = watermarked_frames[n][lt] - original_frames[n][lt]  # the first modified sample
            bethas[n] = watermarked_frames[n][lf - 1] - original_frames[n][lf - 1]  # the last modified sample
            for k in range(lt):
                if k == 0:
                    watermarked_frames[n][k] = original_frames[n][k] + alphas[n] * (k + 1) / (lt + 1)
                else:
                    watermarked_frames[n][k] = original_frames[n][k] + bethas[n - 1] + (alphas[n] - bethas[n - 1]) * (
                            k + 1) / (lt + 1)

        return watermarked_frames

    @staticmethod
    def get_energy(C):
        return np.sum(np.square(C))
