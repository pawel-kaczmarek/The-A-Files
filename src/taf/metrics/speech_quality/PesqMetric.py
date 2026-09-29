from numbers import Number

import numpy as np
import pesq as pypesq  # https://github.com/ludlows/python-pesq

from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


class PesqMetric(Metric):

    card = MetricCard(
        title="Perceptual evaluation of speech quality",
        abbreviation="PESQ",
        category="speech_quality",
        scale="MOS-LQO 1–4.64",
        references=(Reference("ITU-T P.862 / Wang et al.", 2022, doi="10.5281/zenodo.6549559"),),
        summary=Text(
            en="Predicts perceived speech quality by comparing degraded speech with its reference.",
            pl="Przewiduje postrzeganą jakość mowy przez porównanie zniekształconego sygnału z oryginałem.",
        ),
        details=Text(
            en=(
                "Uses the PESQ perceptual model to align and compare auditory representations, "
                "combining disturbance measures into MOS-LQO. Higher is better; the catalogue reports "
                "the wideband scale up to about 4.64. The implementation depends on the PESQ library "
                "and supported speech sample rates. It is a speech-quality estimate, not a general "
                "music metric or a bit-error measure. Input must be sampled at 8 or 16 kHz. Two "
                "components are reported: the raw narrowband P.862 score (undefined, and reported as "
                "missing, at 16 kHz) and MOS-LQO."
            ),
            pl=(
                "Model percepcyjny PESQ wyrównuje i porównuje reprezentacje słuchowe, łącząc miary "
                "zakłóceń w MOS-LQO. Wyższy wynik jest lepszy; katalog podaje skalę szerokopasmową do "
                "około 4,64. Implementacja zależy od biblioteki PESQ i obsługiwanych częstotliwości "
                "mowy. To estymacja jakości mowy, a nie ogólna miara muzyki ani błędów bitowych. "
                "Wejście musi mieć częstotliwość 8 lub 16 kHz. Zwracane są dwie składowe: surowy wynik "
                "wąskopasmowy P.862 (niezdefiniowany i raportowany jako brak przy 16 kHz) oraz MOS-LQO."
            ),
        ),
    )

    higher_is_better = True
    # Raw P.862 score (narrowband only; NaN at 16 kHz) and MOS-LQO.
    components = ("p862_raw", "mos_lqo")

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
