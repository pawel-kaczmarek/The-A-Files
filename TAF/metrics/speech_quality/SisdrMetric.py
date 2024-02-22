from numbers import Number

import numpy as np

from TAF.models.Metric import Metric


class SisdrMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        eps = np.finfo(samples_original.dtype).eps
        reference = samples_processed.reshape(samples_processed.size, 1)
        estimate = samples_original.reshape(samples_original.size, 1)
        Rss = np.dot(reference.T, reference)

        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true ** 2).sum()
        Snn = (e_res ** 2).sum()

        return np.array([10 * np.log10((eps + Sss) / (eps + Snn))])

    def name(self) -> str:
        return "Scale-invariant SDR (SISDR)"
