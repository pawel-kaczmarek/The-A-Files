from numbers import Number

import numpy as np
import pystoi as pystoi  # https://github.com/mpariente/pystoi

from TAF.models.Metric import Metric


class StoiMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float,
                  overlap: float) -> Number| np.ndarray:
        return pystoi.stoi(samples_original, samples_processed, fs)

    def name(self) -> str:
        return "Short-time objective intelligibility (STOI)"
