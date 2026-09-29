"""Mean relative STFT magnitude error at 16, 32 and 64 ms resolutions."""

import numpy as np

from taf.metrics.common.spectral import aligned_mono, magnitude
from taf.models.Metric import Metric
from taf.models.card import MetricCard, Text


class SpectralConvergenceMetric(Metric):

    card = MetricCard(
        title="Multi-resolution spectral convergence",
        abbreviation="MRSC",
        category="speech_quality",
        scale="ratio",
        domain="audio",
        summary=Text(
            en="Measures relative spectral error at several time-frequency resolutions.",
            pl="Mierzy względny błąd widma w kilku rozdzielczościach czasowo-częstotliwościowych.",
        ),
        details=Text(
            en=(
                "Computes STFT magnitudes with 16, 32 and 64 ms windows and 75% overlap. At each "
                "resolution it divides the Frobenius norm of the magnitude error by the reference norm, "
                "then averages. Lower is better and zero indicates matching magnitudes. A silent "
                "reference is undefined; some phase changes remain invisible. This ratio is not a "
                "perceptual MOS."
            ),
            pl=(
                "Liczy moduły STFT z oknami 16, 32 i 64 ms oraz nakładaniem 75%. Dla każdej "
                "rozdzielczości dzieli normę Frobeniusa błędu modułu przez normę oryginału i uśrednia "
                "wyniki. Mniej jest lepiej, a zero oznacza zgodne moduły. Dla cichego odniesienia wynik "
                "jest nieokreślony; część zmian fazy pozostaje niewidoczna. Ten iloraz nie jest "
                "percepcyjnym MOS."
            ),
        ),
    )

    higher_is_better = False

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        x, y = aligned_mono(samples_original, samples_processed, fs)
        errors = []
        for seconds in (0.016, 0.032, 0.064):
            a, b = magnitude(x, fs, seconds), magnitude(y, fs, seconds)
            denominator = np.linalg.norm(a)
            if denominator == 0:
                raise ValueError("Spectral convergence is undefined for a silent reference.")
            errors.append(np.linalg.norm(a - b) / denominator)
        return float(np.mean(errors))

    def name(self):
        return "Multi-resolution spectral convergence (MRSC)"
