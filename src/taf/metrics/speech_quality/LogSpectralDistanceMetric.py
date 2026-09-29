"""Mean framewise RMS log-magnitude distance; diagnostic, not a MOS predictor."""

import numpy as np

from taf.metrics.common.spectral import aligned_mono, magnitude
from taf.models.Metric import Metric
from taf.models.card import MetricCard, Text


class LogSpectralDistanceMetric(Metric):

    card = MetricCard(
        title="Log-spectral distance",
        abbreviation="LSD",
        category="speech_quality",
        scale="dB",
        domain="audio",
        summary=Text(
            en="Measures the difference between short-time log-magnitude spectra in decibels.",
            pl="Mierzy różnicę krótkoczasowych logarytmicznych widm amplitudowych w decybelach.",
        ),
        details=Text(
            en=(
                "Uses 32 ms Hann windows with 75% overlap and a −100 dBFS magnitude floor. For each "
                "frame it takes the RMS spectral difference in dB, then averages frames. Lower is "
                "closer to the reference. It exposes spectral changes but ignores phase differences "
                "that leave magnitudes unchanged; it is a diagnostic, not a MOS predictor."
            ),
            pl=(
                "Używa okien Hanna 32 ms z nakładaniem 75% i dolnym ograniczeniem modułu −100 dBFS. Dla "
                "ramki liczy RMS różnic widma w dB, a następnie uśrednia ramki. Mniej oznacza większą "
                "zgodność z oryginałem. Ujawnia zmiany widmowe, ale pomija zmiany fazy bez zmiany "
                "modułu; to diagnostyka, nie predykcja MOS."
            ),
        ),
    )

    higher_is_better = False

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        x, y = aligned_mono(samples_original, samples_processed, fs)
        # Fixed 32 ms analysis and absolute -100 dB magnitude floor relative to
        # digital full scale. Equal weighting of bins including DC and Nyquist.
        a, b = magnitude(x, fs, 0.032), magnitude(y, fs, 0.032)
        delta = 20 * np.log10(np.maximum(a, 1e-5)) - 20 * np.log10(np.maximum(b, 1e-5))
        return float(np.sqrt(np.mean(delta ** 2, axis=0)).mean())

    def name(self):
        return "Log-spectral distance (LSD)"
