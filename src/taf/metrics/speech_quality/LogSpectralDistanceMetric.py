"""Mean framewise RMS log-magnitude distance; diagnostic, not a MOS predictor."""

import numpy as np

from taf.metrics.common.spectral import aligned_mono, magnitude
from taf.models.Metric import Metric


class LogSpectralDistanceMetric(Metric):
    higher_is_better = False

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        x, y = aligned_mono(samples_original, samples_processed, fs)
        # Fixed 32 ms analysis and absolute -100 dB magnitude floor relative to
        # digital full scale. Equal weighting of bins including DC and Nyquist.
        a, b = magnitude(x, fs, 0.032), magnitude(y, fs, 0.032)
        delta = 20 * np.log10(np.maximum(a, 1e-5)) - 20 * np.log10(np.maximum(b, 1e-5))
        return float(np.sqrt(np.mean(delta ** 2, axis=0)).mean())

    def name(self):
        return "Log-spectral distance (LSD)"
