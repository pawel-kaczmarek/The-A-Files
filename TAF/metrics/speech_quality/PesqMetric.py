from numbers import Number

import numpy as np
import pesq as pypesq  # https://github.com/ludlows/python-pesq

from TAF.models.Metric import Metric


class PesqMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        if fs == 8000:
            mos_lqo = pypesq.pesq(fs, samples_original, samples_processed, 'nb')
            # 0.999 + ( 4.999-0.999 ) / ( 1+np.exp(-1.4945*pesq_mos+4.6607) )
            pesq_mos = 46607 / 14945 - (2000 * np.log(1 / (mos_lqo / 4 - 999 / 4000) - 1)) / 2989
        elif fs == 16000:
            mos_lqo = pypesq.pesq(fs, samples_original, samples_processed, 'wb')
            pesq_mos = np.NaN
        else:
            raise ValueError('fs must be either 8 kHz or 16 kHz')

        return np.array([pesq_mos, mos_lqo])

    def name(self) -> str:
        return "Perceptual Evaluation of Speech Quality (PESQ)"
