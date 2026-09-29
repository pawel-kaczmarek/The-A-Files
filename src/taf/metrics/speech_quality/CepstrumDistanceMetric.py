from numbers import Number

import numpy as np

from taf.models.Metric import Metric
from taf.metrics.common.metrics_helper import extract_overlapped_windows, lpcoeff, lpc2cep
from taf.models.card import MetricCard, Reference, Text


class CepstrumDistanceMetric(Metric):

    card = MetricCard(
        title="Cepstral distance",
        abbreviation="CD",
        category="speech_quality",
        scale="dB",
        references=(Reference("Loizou", 2013, doi="10.1201/b14529"),),
        summary=Text(
            en="Compares cepstral representations of reference and processed speech.",
            pl="Porównuje reprezentacje cepstralne oryginalnej i przetworzonej mowy.",
        ),
        details=Text(
            en=(
                "Derives cepstral coefficients from short-frame speech models and computes their "
                "distance. Cepstral features describe spectral-envelope shape rather than individual "
                "waveform samples. Lower distance in dB means a closer envelope. It needs a reference "
                "and should be interpreted alongside other metrics, since matching envelopes do not "
                "imply identical phase or perceptual quality."
            ),
            pl=(
                "Wyznacza współczynniki cepstralne z modeli krótkich ramek mowy i oblicza ich "
                "odległość. Cechy cepstralne opisują obwiednię widma zamiast pojedynczych próbek. "
                "Mniejsza odległość w dB oznacza bliższą obwiednię. Wymaga odniesienia i interpretacji "
                "z innymi metrykami, bo podobne obwiednie nie gwarantują zgodnej fazy ani jakości "
                "słuchowej."
            ),
        ),
    )

    higher_is_better = False

    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:

        clean_length = len(samples_original)
        processed_length = len(samples_processed)

        winlength = round(frame_len * fs)  # window length in samples
        skiprate = int(np.floor((1 - overlap) * frame_len * fs))  # window skip in samples

        if fs < 10000:
            P = 10  # LPC Analysis Order
        else:
            P = 16;  # this could vary depending on sampling frequency.

        C = 10 * np.sqrt(2) / np.log(10)

        numFrames = int(clean_length / skiprate - (winlength / skiprate));  # number of frames

        hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
        samples_original_framed = extract_overlapped_windows(
            samples_original[0:int(numFrames) * skiprate + int(winlength - skiprate)], winlength, winlength - skiprate,
            hannWin)
        samples_processed_framed = extract_overlapped_windows(
            samples_processed[0:int(numFrames) * skiprate + int(winlength - skiprate)], winlength, winlength - skiprate,
            hannWin)
        distortion = np.zeros((numFrames,))

        for ii in range(numFrames):
            A_clean, R_clean = lpcoeff(samples_original_framed[ii, :], P)
            A_proc, R_proc = lpcoeff(samples_processed_framed[ii, :], P)

            C_clean = lpc2cep(A_clean)
            C_processed = lpc2cep(A_proc)
            distortion[ii] = min((10, C * np.linalg.norm(C_clean - C_processed)))

        IS_dist = distortion
        alpha = 0.95
        IS_len = round(len(IS_dist) * alpha)
        IS = np.sort(IS_dist)
        cep_mean = np.mean(IS[0: IS_len])
        return cep_mean

    def name(self) -> str:
        return "Cepstrum Distance Objective Speech Quality Measure (CD)"
