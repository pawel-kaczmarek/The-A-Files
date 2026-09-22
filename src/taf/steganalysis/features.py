"""Feature extraction for audio steganalysis.

The features follow the shape of the residual-and-co-occurrence descriptors
that rich-model steganalysis is built on: embedding perturbs the relation
between neighbouring samples, so a prediction residual exposes it far better
than the waveform itself. Two families are computed and concatenated:

* Markov transition probabilities of the quantised, truncated residual, in
  both directions. These respond to sample-domain embedding (LSB and its
  relatives), which leaves a characteristic footprint in the fine structure.
* Log-spectral band statistics. These respond to transform-domain and
  spread-spectrum embedding, which barely touches neighbouring-sample
  relations but does tilt the spectrum.
"""
from __future__ import annotations

import numpy as np

_TRUNCATION = 3
_BAND_COUNT = 16


def _residual(audio: np.ndarray) -> np.ndarray:
    """Second-order prediction residual of the waveform."""
    return audio[:-2] - 2.0 * audio[1:-1] + audio[2:]


def _quantized_residual(audio: np.ndarray, quantization: float) -> np.ndarray:
    """Quantise and truncate, so the alphabet stays small enough to count."""
    scale = quantization * max(float(np.std(audio)), 1e-12)
    quantized = np.rint(_residual(audio) / scale)
    # A non-finite sample would survive the clip and become a garbage index
    # after the cast, so it is flattened to zero first.
    quantized = np.nan_to_num(quantized, nan=0.0, posinf=_TRUNCATION, neginf=-_TRUNCATION)
    return np.clip(quantized, -_TRUNCATION, _TRUNCATION).astype(np.int64)


def _markov_features(audio: np.ndarray, quantization: float) -> np.ndarray:
    """Transition probabilities between consecutive residual values."""
    values = _quantized_residual(audio, quantization) + _TRUNCATION
    alphabet = 2 * _TRUNCATION + 1

    counts = np.zeros((alphabet, alphabet), dtype=np.float64)
    np.add.at(counts, (values[:-1], values[1:]), 1.0)

    # Row-normalise into conditional probabilities: this makes the descriptor
    # independent of signal length, so clips of different durations compare.
    row_sums = counts.sum(axis=1, keepdims=True)
    forward = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    column_sums = counts.sum(axis=0, keepdims=True)
    backward = np.divide(counts, column_sums, out=np.zeros_like(counts), where=column_sums > 0)

    return np.concatenate([forward.ravel(), backward.ravel()])


def _spectral_features(audio: np.ndarray, frame_length: int = 1024) -> np.ndarray:
    """Mean and spread of the log spectrum in equal-width bands."""
    usable = (len(audio) // frame_length) * frame_length
    if usable < frame_length:
        raise ValueError("signal is too short for spectral features")

    frames = audio[:usable].reshape(-1, frame_length)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1))
    log_spectrum = np.log(spectrum + 1e-10)

    bands = np.array_split(log_spectrum, _BAND_COUNT, axis=1)
    means = np.array([float(np.mean(band)) for band in bands])
    deviations = np.array([float(np.std(band)) for band in bands])

    # Centre the band means: an overall level change is a volume difference,
    # not evidence of embedding, and leaving it in lets the classifier learn
    # loudness instead of steganography.
    return np.concatenate([means - float(np.mean(means)), deviations])


def extract_features(audio: np.ndarray, quantizations: tuple = (1.0, 2.0)) -> np.ndarray:
    """Return the steganalysis feature vector of one signal.

    Args:
        audio: Mono waveform.
        quantizations: Residual quantisation steps, as multiples of the signal
            standard deviation. Several steps are used because embedding
            strengths differ by orders of magnitude between methods, and a
            single step is only sensitive around its own scale.
    """
    audio = np.asarray(audio, dtype=np.float64).ravel()
    if audio.size < 8:
        raise ValueError("signal is too short for steganalysis features")

    parts = [_markov_features(audio, quantization) for quantization in quantizations]
    parts.append(_spectral_features(audio))
    features = np.concatenate(parts)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def feature_count(quantizations: tuple = (1.0, 2.0)) -> int:
    """Length of the vector extract_features returns."""
    alphabet = 2 * _TRUNCATION + 1
    return len(quantizations) * 2 * alphabet ** 2 + 2 * _BAND_COUNT
