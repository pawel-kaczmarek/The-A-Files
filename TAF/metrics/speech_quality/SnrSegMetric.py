from numbers import Number

import numpy as np

from TAF.models.Metric import Metric
from TAF.metrics.common.metrics_helper import extract_overlapped_windows


class SnrSegMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        eps = np.finfo(np.float64).eps

        winlength = round(frame_len * fs)  # window length in samples
        skiprate = int(np.floor((1 - overlap) * frame_len * fs))  # window skip in samples
        MIN_SNR = -10  # minimum SNR in dB
        MAX_SNR = 35  # maximum SNR in dB

        hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
        clean_speech_framed = extract_overlapped_windows(samples_original, winlength, winlength - skiprate, hannWin)
        processed_speech_framed = extract_overlapped_windows(samples_processed, winlength, winlength - skiprate, hannWin)

        signal_energy = np.power(clean_speech_framed, 2).sum(-1)
        noise_energy = np.power(clean_speech_framed - processed_speech_framed, 2).sum(-1)

        segmental_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
        segmental_snr[segmental_snr < MIN_SNR] = MIN_SNR
        segmental_snr[segmental_snr > MAX_SNR] = MAX_SNR
        segmental_snr = segmental_snr[:-1]  # remove last frame -> not valid
        return np.mean(segmental_snr)

    def name(self) -> str:
        return "Segmental Signal-to-Noise Ratio (SNRseg)"
