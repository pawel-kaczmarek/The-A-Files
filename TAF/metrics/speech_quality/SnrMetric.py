from numbers import Number

import numpy as np

from TAF.models.Metric import Metric


class SnrMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        nominator = np.sum(samples_original ** 2)
        denominator = np.sum((samples_original - samples_processed) ** 2)
        # if denominator == 0:
        #     return ValueError("Max SNR value! Signals are identical.")
        return 10 * np.log10(nominator / denominator)

    def name(self) -> str:
        return "Signal-to-Noise Ratio (SNR)"
