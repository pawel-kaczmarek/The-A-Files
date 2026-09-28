"""Extended STOI using the existing pystoi dependency (Jensen & Taal, 2016)."""

import warnings
import numpy as np

from pystoi import stoi

from taf.metrics.common.spectral import aligned_mono
from taf.models.Metric import Metric


class EstoiMetric(Metric):
    higher_is_better = True

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        x, y = aligned_mono(samples_original, samples_processed, fs)
        if not np.any(x):
            raise ValueError("eSTOI requires a non-silent speech reference.")
        # pystoi returns a small sentinel with a warning for insufficient active
        # speech. Record a metric error rather than presenting it as a score.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning, module="pystoi.*")
            return float(stoi(x, y, fs, extended=True))

    def name(self):
        return "Extended short-time objective intelligibility (eSTOI)"
