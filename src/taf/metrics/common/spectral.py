"""Explicit, sample-aligned spectral analysis shared by diagnostic metrics."""

import numpy as np
from scipy.signal import stft


def aligned_mono(original, processed, fs):
    x, y = np.asarray(original, dtype=np.float64), np.asarray(processed, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("Expected equal-length, nonempty mono signals (at least two samples).")
    if fs <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Signals must be finite and sample rate positive.")
    return x, y


def magnitude(samples, fs, seconds):
    window = min(len(samples), max(2, round(fs * seconds)))
    # Hann, 75% overlap, no centering or trailing padding. No gain or delay fit.
    return np.abs(stft(samples, fs=fs, window="hann", nperseg=window,
                       noverlap=window * 3 // 4, boundary=None, padded=False)[2])
