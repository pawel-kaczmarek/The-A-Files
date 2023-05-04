from numbers import Number

import numpy as np
from srmrpy import srmr

from TAF.models.Metric import Metric


class SrmrMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        ratio, energy = srmr(samples_original, fs)
        ratio_processed, energy_processed = srmr(samples_processed, fs)
        # The outputs are ratio, which is the SRMR scorez
        # and energy, a 3D matrix with the per-frame modulation spectrum extracted from the input.
        return np.array([ratio, ratio_processed])

    def name(self) -> str:
        return "Speech-to-reverberation modulation energy ratio (SRMR)"
