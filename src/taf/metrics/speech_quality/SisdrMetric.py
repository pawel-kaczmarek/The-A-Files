from numbers import Number

import numpy as np

from taf.models.Metric import Metric
from taf.models.card import MetricCard, Reference, Text


class SisdrMetric(Metric):

    card = MetricCard(
        title="Scale-invariant signal-to-distortion ratio",
        abbreviation="SI-SDR",
        category="speech_quality",
        scale="dB",
        domain="audio",
        references=(Reference("Le Roux et al.", 2019, doi="10.1109/ICASSP.2019.8683855"),),
        summary=Text(
            en="Measures waveform distortion after compensating for a single global scale factor.",
            pl="Mierzy zniekształcenie przebiegu po kompensacji jednego globalnego współczynnika skali.",
        ),
        details=Text(
            en=(
                "Projects the processed signal onto the reference and compares projected target energy "
                "with residual energy in dB. Higher SI-SDR is better. Unlike ordinary SNR, a uniform "
                "gain change is discounted; timing errors and other waveform changes still count. Use "
                "alongside level-sensitive metrics when amplitude preservation matters."
            ),
            pl=(
                "Rzutuje sygnał przetworzony na oryginał i porównuje energię projekcji z energią reszty "
                "w dB. Wyższy SI-SDR jest lepszy. W odróżnieniu od zwykłego SNR pomija jednolite "
                "wzmocnienie; błędy czasu i inne zmiany przebiegu nadal wpływają na wynik. Jeśli ważne "
                "jest zachowanie amplitudy, warto zestawić go z miarą czułą na poziom."
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
        eps = np.finfo(samples_original.dtype).eps
        reference = samples_processed.reshape(samples_processed.size, 1)
        estimate = samples_original.reshape(samples_original.size, 1)
        Rss = np.dot(reference.T, reference)

        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true ** 2).sum()
        Snn = (e_res ** 2).sum()

        return np.array([10 * np.log10((eps + Sss) / (eps + Snn))])

    def name(self) -> str:
        return "Scale-invariant SDR (SISDR)"
