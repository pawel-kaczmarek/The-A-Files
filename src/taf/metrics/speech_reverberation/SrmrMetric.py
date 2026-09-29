from numbers import Number

import numpy as np

from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


class SrmrMetric(Metric):

    card = MetricCard(
        title="Speech-to-reverberation modulation energy ratio",
        abbreviation="SRMR",
        category="speech_reverberation",
        scale="ratio",
        intrusive=False,
        references=(Reference("Falk et al.", 2010, doi="10.1109/TASL.2010.2052247"),),
        summary=Text(
            en="Estimates reverberation-related degradation from the processed signal alone.",
            pl="Szacuje degradację związaną z pogłosem na podstawie samego sygnału przetworzonego.",
        ),
        details=Text(
            en=(
                "Analyses modulation energy in auditory-band envelopes and forms a ratio of lower to "
                "higher modulation-frequency energy. Higher SRMR generally suggests less reverberant "
                "degradation for speech. It does not require a clean reference and is not a direct RT60 "
                "estimate. Noise, speaking style and signal content can also influence its value."
            ),
            pl=(
                "Analizuje energię modulacji obwiedni w pasmach słuchowych i tworzy stosunek energii "
                "niższych do wyższych częstotliwości modulacji. Wyższe SRMR zwykle sugeruje mniejszą "
                "degradację pogłosową mowy. Nie wymaga czystego oryginału i nie jest bezpośrednią "
                "estymacją RT60. Na wynik wpływają też szum, sposób mówienia i treść sygnału."
            ),
        ),
    )

    higher_is_better = True
    # SRMR is non-intrusive: the score of the cover is a reference, only the
    # score of the processed signal describes it.
    components = ("cover", "processed")
    component_directions = (None, True)

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        from taf.metrics.speech_reverberation.srmrpy import srmr

        ratio, energy = srmr(samples_original, fs)
        ratio_processed, energy_processed = srmr(samples_processed, fs)
        # The outputs are ratio, which is the SRMR scorez
        # and energy, a 3D matrix with the per-frame modulation spectrum extracted from the input.
        return np.array([ratio, ratio_processed])

    def name(self) -> str:
        return "Speech-to-reverberation modulation energy ratio (SRMR)"
