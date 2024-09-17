from numbers import Number

import numpy as np
from museval import metrics  # https://github.com/sigsep/sigsep-mus-eval

from TAF.models.Metric import Metric


class BSSEvalMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        result = metrics.bss_eval(reference_sources=samples_original,  # shape: [nsrc, nsample, nchannels]
                                  estimated_sources=samples_processed)
        values = [item[0][0] for item in result]
        return np.asarray(values) # (sdr, isr, sir, sar, perm)

    def name(self) -> str:
        return "BSS_EVAL version 4."
