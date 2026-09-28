"""Mean relative STFT magnitude error at 16, 32 and 64 ms resolutions."""

import numpy as np

from taf.metrics.common.spectral import aligned_mono, magnitude
from taf.models.Metric import Metric


class SpectralConvergenceMetric(Metric):
    higher_is_better = False

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        x, y = aligned_mono(samples_original, samples_processed, fs)
        errors = []
        for seconds in (0.016, 0.032, 0.064):
            a, b = magnitude(x, fs, seconds), magnitude(y, fs, seconds)
            denominator = np.linalg.norm(a)
            if denominator == 0:
                raise ValueError("Spectral convergence is undefined for a silent reference.")
            errors.append(np.linalg.norm(a - b) / denominator)
        return float(np.mean(errors))

    def name(self):
        return "Multi-resolution spectral convergence (MRSC)"
