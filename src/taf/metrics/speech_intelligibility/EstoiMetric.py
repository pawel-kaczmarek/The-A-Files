"""Extended STOI using the existing pystoi dependency (Jensen & Taal, 2016)."""

import warnings
import numpy as np

from pystoi import stoi

from taf.metrics.common.spectral import aligned_mono
from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


class EstoiMetric(Metric):

    card = MetricCard(
        title="Extended short-time objective intelligibility",
        abbreviation="eSTOI",
        category="speech_intelligibility",
        scale="index (usually 0–1)",
        references=(Reference("Jensen & Taal", 2016, doi="10.1109/TASLP.2016.2585878"),),
        summary=Text(
            en="Extends STOI with a spectro-temporal comparison useful for fluctuating interference.",
            pl="Rozszerza STOI o porównanie czasowo-widmowe przydatne dla zmiennych zakłóceń.",
        ),
        details=Text(
            en=(
                "Uses pystoi in extended mode, comparing normalised short-time time-frequency envelope "
                "patterns. Higher eSTOI indicates better predicted speech intelligibility, usually on "
                "an index scale near 0–1. It requires reference speech and remains a proxy: it does not "
                "measure listening quality, message decoding or steganographic security."
            ),
            pl=(
                "Korzysta z rozszerzonego trybu pystoi i porównuje znormalizowane krótkoczasowe wzorce "
                "obwiedni czasowo-częstotliwościowych. Wyższe eSTOI oznacza lepszą przewidywaną "
                "zrozumiałość mowy, zwykle w skali bliskiej 0–1. Wymaga mowy odniesienia i pozostaje "
                "wskaźnikiem zastępczym: nie mierzy jakości odsłuchu, odczytu danych ani bezpieczeństwa "
                "steganografii."
            ),
        ),
    )

    higher_is_better = True

    def calculate(self, samples_original, samples_processed, fs, frame_len=0.03, overlap=0.75):
        x, y = aligned_mono(samples_original, samples_processed, fs)
        if not np.any(x):
            raise ValueError("eSTOI requires a non-silent speech reference.")
        # pystoi returns a small sentinel with a warning for insufficient active
        # speech. Record a metric error rather than presenting it as a score.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning, module="pystoi.*")
            return float(stoi(x, y, fs, extended=True))

    def name(self):
        return "Extended short-time objective intelligibility (eSTOI)"
