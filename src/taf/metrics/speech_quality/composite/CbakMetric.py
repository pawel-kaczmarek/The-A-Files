from numbers import Number

import numpy as np

from taf.metrics.speech_quality.composite.CompositeSpeechEnhancementMetric import BaseSpeechEnhancementMetric
from taf.models.card import MetricCard, Reference, Text


class CbakMetric(BaseSpeechEnhancementMetric):

    card = MetricCard(
        title="Composite background-noise rating",
        abbreviation="Cbak",
        category="speech_quality",
        scale="1–5",
        references=(Reference("Hu & Loizou", 2008, doi="10.1109/TASL.2007.911054"),),
        summary=Text(
            en="Predicts the perceived intrusiveness of background noise in processed speech.",
            pl="Przewiduje dokuczliwość szumu tła w przetworzonej mowie.",
        ),
        details=Text(
            en=(
                "Combines PESQ, segmental SNR and weighted spectral slope through an empirical "
                "regression. The resulting composite score is limited to the 1–5 range, with higher "
                "values meaning less intrusive background noise. It is calibrated for speech "
                "enhancement and needs the original reference; it is not a direct measurement of "
                "background-noise power."
            ),
            pl=(
                "Łączy PESQ, segmentowy SNR i ważone nachylenie widma przez regresję empiryczną. Wynik "
                "złożony jest ograniczany do zakresu 1–5; więcej oznacza mniej dokuczliwe tło. Model "
                "skalibrowano dla poprawy jakości mowy i wymaga on oryginału; nie jest bezpośrednim "
                "pomiarem mocy szumu."
            ),
        ),
    )

    higher_is_better = True

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        metrics = super().calculate_internal(samples_original=samples_original,
                                             samples_processed=samples_processed,
                                             fs=fs)
        return metrics[1]

    def name(self) -> str:
        return "CBAK: Predicted rating of background distortion"
