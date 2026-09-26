from numbers import Number

import numpy as np
import pystoi as pystoi  # https://github.com/mpariente/pystoi

from taf.models.Metric import Metric


class StoiMetric(Metric):
    higher_is_better = True

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number| np.ndarray:
        return pystoi.stoi(samples_original, samples_processed, fs)

    def name(self) -> str:
        return "Short-time objective intelligibility (STOI)"
