from numbers import Number

import numpy as np
import pystoi as pystoi  # https://github.com/mpariente/pystoi

from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


class StoiMetric(Metric):

    card = MetricCard(
        title="Short-time objective intelligibility",
        abbreviation="STOI",
        category="speech_intelligibility",
        scale="0–1",
        references=(Reference("Taal et al.", 2010, doi="10.1109/ICASSP.2010.5495701"),),
        summary=Text(
            en="Estimates speech intelligibility from preservation of short-time spectral envelopes.",
            pl="Szacuje zrozumiałość mowy na podstawie zachowania krótkoczasowych obwiedni widmowych.",
        ),
        details=Text(
            en=(
                "Compares reference and processed temporal envelopes in one-third-octave bands over "
                "short segments, after the algorithm’s normalisation and clipping steps. Higher values, "
                "commonly near the 0–1 range, suggest better intelligibility. It does not recognise "
                "words: the score is not a percentage of correctly understood words and is not a "
                "general music-quality metric."
            ),
            pl=(
                "Porównuje obwiednie czasowe oryginału i wyniku w pasmach tercjowych oraz krótkich "
                "segmentach, po normalizacji i ograniczaniu przewidzianym w algorytmie. Wyższy wynik, "
                "zwykle w pobliżu zakresu 0–1, sugeruje lepszą zrozumiałość. Nie rozpoznaje słów: wynik "
                "nie jest procentem zrozumianych wyrazów ani ogólną miarą jakości muzyki."
            ),
        ),
    )

    higher_is_better = True

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number| np.ndarray:
        return pystoi.stoi(samples_original, samples_processed, fs)

    def name(self) -> str:
        return "Short-time objective intelligibility (STOI)"
