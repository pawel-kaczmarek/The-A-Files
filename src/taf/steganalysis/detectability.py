"""Measure how detectable a steganography method is.

Quality metrics say how much a method damages the audio and the robustness
attacks say how much survives, but neither answers the question a
steganographer actually cares about: can an observer tell that anything was
embedded at all? This module answers it by training a steganalyser to
separate covers from the stego signals a method produces, and reporting how
well it does on held-out material.

An accuracy near 0.5 means the classifier is guessing, so the method is
undetectable by these features; 1.0 means it is trivially detectable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from taf.models.SteganographyMethod import SteganographyMethod
from taf.steganalysis.EnsembleClassifier import EnsembleClassifier
from taf.steganalysis.features import extract_features


@dataclass(frozen=True)
class DetectabilityResult:
    """Outcome of a steganalysis run against one method."""

    method: str
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    train_size: int
    test_size: int
    out_of_bag_error: Optional[float]
    #: 95% Wilson interval of the test accuracy.
    accuracy_ci95_low: Optional[float] = None
    accuracy_ci95_high: Optional[float] = None
    #: One-sided exact binomial test of "accuracy > 0.5" on the test set.
    p_value: Optional[float] = None

    @property
    def undetectable(self) -> bool:
        """True when the point estimate is within 0.05 of a coin toss.

        A heuristic on the point estimate only; with a small test set an
        accuracy of 0.6 can still be chance. ``significantly_detectable``
        accounts for the size of the test set.
        """
        return self.accuracy <= 0.55

    @property
    def significantly_detectable(self) -> bool:
        """True when the detector beats chance at the 5% level."""
        return self.p_value is not None and self.p_value < 0.05


def _windows(audio: np.ndarray, window_length: int, stride: Optional[int] = None) -> List[np.ndarray]:
    stride = stride or window_length
    return [
        audio[start:start + window_length]
        for start in range(0, len(audio) - window_length + 1, stride)
    ]


def measure_detectability(
    method: SteganographyMethod,
    covers: Sequence[np.ndarray],
    message_length: int = 20,
    window_length: int = 32000,
    test_fraction: float = 0.3,
    seed: int = 20240521,
    classifier: Optional[EnsembleClassifier] = None,
) -> DetectabilityResult:
    """Train a steganalyser on one method and score it on held-out windows.

    Args:
        method: The method under test.
        covers: Cover signals. They are cut into windows, so a handful of
            recordings is enough to build a training set.
        message_length: Payload embedded in every window.
        window_length: Samples per window, and therefore per example.
        test_fraction: Share of examples held out for scoring.
        seed: Seed for the train/test split.
        classifier: Pre-configured ensemble; a default one is used otherwise.

    Each window is embedded with its own random message, so the classifier
    learns the footprint of the method rather than of one particular payload.
    """
    rng = np.random.default_rng(seed)
    cover_windows: List[np.ndarray] = []
    for cover in covers:
        cover_windows.extend(_windows(np.asarray(cover, dtype=np.float64), window_length))

    if len(cover_windows) < 4:
        raise ValueError("not enough cover material: need at least 4 windows")

    features: List[np.ndarray] = []
    labels: List[int] = []
    for window in cover_windows:
        message = [int(bit) for bit in rng.integers(0, 2, message_length)]
        stego = np.asarray(method.encode(window.copy(), message), dtype=np.float64)

        features.append(extract_features(window))
        labels.append(0)
        features.append(extract_features(stego))
        labels.append(1)

    feature_matrix = np.vstack(features)
    label_vector = np.asarray(labels, dtype=np.int64)

    # Split by window, keeping each cover/stego pair together: the same audio
    # on both sides of the split would let the classifier recognise the
    # recording instead of the embedding.
    window_count = len(cover_windows)
    order = rng.permutation(window_count)
    test_count = max(1, int(round(window_count * test_fraction)))
    test_windows, train_windows = order[:test_count], order[test_count:]

    if len(train_windows) < 2:
        raise ValueError("not enough cover material to train and test")

    def rows(window_indices: np.ndarray) -> np.ndarray:
        return np.concatenate([[2 * index, 2 * index + 1] for index in window_indices])

    train_rows, test_rows = rows(train_windows), rows(test_windows)

    model = classifier or EnsembleClassifier()
    model.fit(feature_matrix[train_rows], label_vector[train_rows])

    predictions = model.predict(feature_matrix[test_rows])
    truth = label_vector[test_rows]

    covers_mask = truth == 0
    stego_mask = truth == 1
    correct = int(np.sum(predictions == truth))
    low, high = _wilson_interval(correct, len(truth))

    from scipy.stats import binomtest

    return DetectabilityResult(
        method=method.type(),
        accuracy=correct / len(truth),
        false_positive_rate=float(np.mean(predictions[covers_mask] == 1)),
        false_negative_rate=float(np.mean(predictions[stego_mask] == 0)),
        train_size=len(train_rows),
        test_size=len(test_rows),
        out_of_bag_error=model.oob_error,
        accuracy_ci95_low=low,
        accuracy_ci95_high=high,
        p_value=float(binomtest(correct, len(truth), 0.5, alternative="greater").pvalue),
    )


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0, 1] for small test sets."""
    if trials == 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (proportion + z ** 2 / (2 * trials)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / trials + z ** 2 / (4 * trials ** 2)) / denominator
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))
