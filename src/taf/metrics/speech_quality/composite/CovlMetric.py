from numbers import Number

import numpy as np

from taf.metrics.speech_quality.composite.CompositeSpeechEnhancementMetric import BaseSpeechEnhancementMetric
from taf.models.card import MetricCard, Reference, Text


class CovlMetric(BaseSpeechEnhancementMetric):

    card = MetricCard(
        title="Composite overall-quality rating",
        abbreviation="Covl",
        category="speech_quality",
        scale="1–5",
        references=(Reference("Hu & Loizou", 2008, doi="10.1109/TASL.2007.911054"),),
        summary=Text(
            en="Predicts overall speech quality, combining signal distortion and noise effects.",
            pl="Przewiduje ogólną jakość mowy, łącząc wpływ zniekształcenia sygnału i szumu.",
        ),
        details=Text(
            en=(
                "Uses an empirical combination of PESQ, log-likelihood ratio and weighted spectral "
                "slope to produce a composite 1–5 score. Higher is better. The model was designed for "
                "speech-enhancement evaluation; its scale should not be treated as directly "
                "interchangeable with PESQ, MOSNet or a listening-test MOS."
            ),
            pl=(
                "Używa empirycznego połączenia PESQ, logarytmicznego ilorazu wiarygodności i ważonego "
                "nachylenia widma, tworząc ocenę 1–5. Więcej jest lepiej. Model służy ocenie poprawy "
                "jakości mowy; jego skali nie należy utożsamiać bezpośrednio z PESQ, MOSNet ani MOS z "
                "testu odsłuchowego."
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
        return metrics[2]

    def name(self) -> str:
        return "COVL: Predicted rating of overall quality"
