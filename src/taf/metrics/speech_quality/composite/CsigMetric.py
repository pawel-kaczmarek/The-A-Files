from numbers import Number

import numpy as np

from taf.metrics.speech_quality.composite.CompositeSpeechEnhancementMetric import BaseSpeechEnhancementMetric
from taf.models.card import MetricCard, Reference, Text


class CsigMetric(BaseSpeechEnhancementMetric):

    card = MetricCard(
        title="Composite signal-distortion rating",
        abbreviation="Csig",
        category="speech_quality",
        scale="1–5",
        references=(Reference("Hu & Loizou", 2008, doi="10.1109/TASL.2007.911054"),),
        summary=Text(
            en="Predicts how much the speech signal itself sounds distorted.",
            pl="Przewiduje, jak silnie zniekształcony brzmi sam sygnał mowy.",
        ),
        details=Text(
            en=(
                "Combines PESQ, log-likelihood ratio and weighted spectral slope using an empirical "
                "regression, then limits the prediction to 1–5. Higher values mean better predicted "
                "speech-signal quality. It focuses on speech distortion rather than background-noise "
                "annoyance and is a model estimate, not a listener’s actual rating."
            ),
            pl=(
                "Łączy PESQ, logarytmiczny iloraz wiarygodności i ważone nachylenie widma przez "
                "regresję empiryczną, po czym ogranicza wynik do 1–5. Więcej oznacza lepszą "
                "przewidywaną jakość samej mowy. Skupia się na zniekształceniu mowy, a nie "
                "dokuczliwości tła; jest estymacją modelu, nie rzeczywistą oceną słuchacza."
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
        return metrics[0]

    def name(self) -> str:
        return "CSIG: Predicted rating of speech distortion"
