from numbers import Number

import numpy as np

from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


class SnrMetric(Metric):

    card = MetricCard(
        title="Signal-to-noise ratio",
        abbreviation="SNR",
        category="speech_quality",
        scale="dB",
        domain="audio",
        references=(Reference("Loizou", 2013, doi="10.1201/b14529"),),
        summary=Text(
            en="Measures waveform fidelity as the ratio of original signal energy to error energy.",
            pl="Mierzy wierność przebiegu jako stosunek energii oryginału do energii błędu.",
        ),
        details=Text(
            en=(
                "Computes 10 log10 of reference energy divided by the energy of the samplewise "
                "difference. Higher dB means less error relative to the reference. Requires aligned "
                "signals and penalises gain or time shifts strongly. It measures numerical fidelity, "
                "not perceived quality, message recovery or steganographic detectability."
            ),
            pl=(
                "Oblicza 10 log10 ilorazu energii odniesienia i energii różnicy próbek. Więcej dB "
                "oznacza mniejszy błąd względem oryginału. Wymaga wyrównanych sygnałów i silnie karze "
                "zmianę wzmocnienia lub przesunięcie. Mierzy zgodność liczbową, a nie jakość słuchową, "
                "odczyt wiadomości czy wykrywalność steganografii."
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
        nominator = np.sum(samples_original ** 2)
        denominator = np.sum((samples_original - samples_processed) ** 2)
        # if denominator == 0:
        #     return ValueError("Max SNR value! Signals are identical.")
        return 10 * np.log10(nominator / denominator)

    def name(self) -> str:
        return "Signal-to-Noise Ratio (SNR)"
