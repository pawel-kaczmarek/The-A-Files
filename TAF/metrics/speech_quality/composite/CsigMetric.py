from numbers import Number

import numpy as np

from TAF.metrics.speech_quality.composite.CompositeSpeechEnhancementMetric import BaseSpeechEnhancementMetric


class CsgiMetric(BaseSpeechEnhancementMetric):

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        metrics = super().calculate_internal(samples_original=samples_original,
                                             samples_processed=samples_processed,
                                             fs=fs)
        return metrics[0]

    def name(self) -> str:
        return "CSIG: Predicted rating of speech distortion"
