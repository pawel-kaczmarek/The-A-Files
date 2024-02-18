from numbers import Number

import numpy as np
from loguru import logger

from TAF.metrics.common.metrics_helper import log_mel_spectrogram, sgbfb
from TAF.models.Metric import Metric


class WstmiMetric(Metric):

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        if fs != 10000:
            logger.warning(
                "Speech signals must be time-aligned and of the same length. The algorithms only support 10 KHz sampling rate")
            return -1

        win_length = 25.6
        win_shift = win_length / 2
        freq_range = [64, min(np.floor(fs / 2), 12000)]
        num_bands = 130
        band_factor = 1
        eps = 1e-15
        omega_s = np.array([0.081, 0.128, 0.326, 0.518])
        omega_t = np.array([0, 0.389, 0.619])

        w = [[0.000, 0.031, 0.140],
             [0.013, 0.041, 0.055],
             [0.459, 0.528, 0.000],
             [0.151, 0.000, 0.000]]
        b = 0.16;

        X_spec, _ = log_mel_spectrogram(samples_original, fs=fs, win_shift=win_shift, win_length=win_length,
                                        freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)
        Y_spec, _ = log_mel_spectrogram(samples_processed, fs=fs, win_shift=win_shift, win_length=win_length,
                                        freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)

        X = sgbfb(X_spec, omega_s=omega_s, omega_t=omega_t)
        Y = sgbfb(Y_spec, omega_s=omega_s, omega_t=omega_t)

        X = X + eps * np.random.rand(*X.shape)
        Y = Y + eps * np.random.rand(*Y.shape)

        X = X - np.expand_dims(np.mean(X, axis=3), axis=3)
        Y = Y - np.expand_dims(np.mean(Y, axis=3), axis=3)
        X = X / np.expand_dims(np.sqrt(np.sum(X * X, axis=3)), axis=3)
        Y = Y / np.expand_dims(np.sqrt(np.sum(Y * Y, axis=3)), axis=3)
        rho = np.mean(np.sum(X * Y, axis=3), axis=2)
        d = np.sum(w * rho) + b

        return d

    def name(self) -> str:
        return "Weighted Spectro-Temporal Modulation Index (wSTMI)"
