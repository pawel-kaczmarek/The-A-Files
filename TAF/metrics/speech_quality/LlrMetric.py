from numbers import Number

import numpy as np
from scipy.linalg import toeplitz

from TAF.metrics.common.metrics_helper import extract_overlapped_windows, lpcoeff
from TAF.models.Metric import Metric


class LlrMetric(Metric):
    def calculate(self,
                  samples_original: np.ndarray,
                  samples_processed: np.ndarray,
                  fs: int,
                  frame_len: float = 0.03,
                  overlap: float = 0.75) -> Number | np.ndarray:
        used_for_composite = False  # TODO as param!
        eps = np.finfo(np.float64).eps
        alpha = 0.95
        winlength = round(frame_len * fs)  # window length in samples
        skiprate = int(np.floor((1 - overlap) * frame_len * fs))  # window skip in samples
        if fs < 10000:
            P = 10  # LPC Analysis Order
        else:
            P = 16  # this could vary depending on sampling frequency.

        hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
        clean_speech_framed = extract_overlapped_windows(samples_original + eps, winlength, winlength - skiprate,
                                                         hannWin)
        processed_speech_framed = extract_overlapped_windows(samples_processed + eps, winlength, winlength - skiprate,
                                                             hannWin)
        numFrames = clean_speech_framed.shape[0]
        numerators = np.zeros((numFrames - 1,))
        denominators = np.zeros((numFrames - 1,))

        for ii in range(numFrames - 1):
            A_clean, R_clean = lpcoeff(clean_speech_framed[ii, :], P)
            A_proc, R_proc = lpcoeff(processed_speech_framed[ii, :], P)

            numerators[ii] = A_proc.dot(toeplitz(R_clean).dot(A_proc.T))
            denominators[ii] = A_clean.dot(toeplitz(R_clean).dot(A_clean.T))

        frac = numerators / denominators
        frac[np.isnan(frac)] = np.inf
        frac[frac <= 0] = 1000
        distortion = np.log(frac)
        if not used_for_composite:
            distortion[
                distortion > 2] = 2  # this line is not in composite measure but in llr matlab implementation of loizou
        distortion = np.sort(distortion)
        distortion = distortion[:int(round(len(distortion) * alpha))]
        return np.mean(distortion)

    def name(self) -> str:
        return "Log-likelihood Ratio (LLR)"
