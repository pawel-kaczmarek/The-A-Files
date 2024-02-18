from numbers import Number

import numpy as np
from loguru import logger

from TAF.metrics.common.metrics_helper import log_mel_spectrogram, sgbfb
from TAF.models.Metric import Metric


class StgiMetric(Metric):

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

        STM_channels = np.ones((11, 4))
        thresholds = [[0.252, 0.347, 0.275, 0.189],
                      [0.502, 0.495, 0.404, 0.279],
                      [0.486, 0.444, 0.357, 0.247],
                      [0.456, 0.405, 0.332, 0.229],
                      [0.426, 0.361, 0.287, 0.191],
                      [0.357, 0.299, 0.229, 0.150],
                      [0.269, 0.228, 0.175, 0.114],
                      [0.185, 0.158, 0.118, 0.075],
                      [0.119, 0.103, 0.073, 0.047],
                      [0.081, 0.067, 0.047, 0.030],
                      [0.050, 0.043, 0.031, 0.020]]

        win_length = 25.6
        win_shift = win_length / 2
        freq_range = [64, min(np.floor(fs / 2), 12000)]
        num_bands = 130
        band_factor = 1
        N = 40
        eps = 1e-15

        X, _ = log_mel_spectrogram(samples_original, fs=fs, win_shift=win_shift, win_length=win_length,
                                   freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)
        Y, _ = log_mel_spectrogram(samples_processed, fs=fs, win_shift=win_shift, win_length=win_length,
                                   freq_range=freq_range, num_bands=num_bands, band_factor=band_factor)

        X_hat = sgbfb(X, STM_channels)
        Y_hat = sgbfb(Y, STM_channels)

        intell = np.zeros((X_hat.shape[3] - N + 1,))

        for n in range(N - 1, X_hat.shape[3]):
            X_seg = X_hat[..., (n - N + 1):n + 1]
            Y_seg = Y_hat[..., (n - N + 1):n + 1]

            X_seg = X_seg + eps * np.random.rand(*X_seg.shape)
            Y_seg = Y_seg + eps * np.random.rand(*Y_seg.shape)

            X_seg = X_seg - np.expand_dims(np.mean(X_seg, axis=3), axis=3)
            Y_seg = Y_seg - np.expand_dims(np.mean(Y_seg, axis=3), axis=3)
            X_seg = X_seg / np.expand_dims(np.sqrt(np.sum(X_seg * X_seg, axis=3)), axis=3)
            Y_seg = Y_seg / np.expand_dims(np.sqrt(np.sum(Y_seg * Y_seg, axis=3)), axis=3)

            X_seg = X_seg - np.expand_dims(np.mean(X_seg, axis=2), axis=2)
            Y_seg = Y_seg - np.expand_dims(np.mean(Y_seg, axis=2), axis=2)
            X_seg = X_seg / np.expand_dims(np.sqrt(np.sum(X_seg * X_seg, axis=2)), axis=2)
            Y_seg = Y_seg / np.expand_dims(np.sqrt(np.sum(Y_seg * Y_seg, axis=2)), axis=2)

            d = np.squeeze(np.sum(X_seg * Y_seg, axis=2));
            d = np.squeeze(np.mean(d, axis=2));
            g = d > thresholds
            intell[n - N + 1] = np.mean(g[:])

        rho = np.mean(intell[:])
        return rho

    def name(self) -> str:
        return "Spectro-Temporal Glimpsing Index (STGI)"
