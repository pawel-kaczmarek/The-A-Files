"""Ensemble classifier for steganalysis.

Reference:
    Kodovsky, J., Fridrich, J., & Holub, V. (2012). "Ensemble Classifiers for
    Steganalysis of Digital Media." IEEE Transactions on Information Forensics
    and Security, 7(2), 432-444. https://doi.org/10.1109/TIFS.2011.2175919

Steganalysis feature vectors are high-dimensional and the training sets are
small, which is exactly where a single discriminant overfits. The ensemble
trains many Fisher linear discriminants, each on a random subspace of the
features and a bootstrap sample of the training set, and votes. The subspace
width is chosen on out-of-bag error, so no separate validation split is
needed.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


class _FisherDiscriminant:
    """Two-class Fisher linear discriminant with a regularised scatter."""

    def __init__(self, regularization: float = 1e-8):
        self.regularization = regularization
        self.weights: Optional[np.ndarray] = None
        self.threshold: float = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_FisherDiscriminant":
        cover = features[labels == 0]
        stego = features[labels == 1]
        if len(cover) == 0 or len(stego) == 0:
            raise ValueError("both classes must be present in the training set")

        cover_mean, stego_mean = cover.mean(axis=0), stego.mean(axis=0)
        scatter = np.cov(cover, rowvar=False) + np.cov(stego, rowvar=False)
        scatter = np.atleast_2d(scatter)
        scatter += np.eye(scatter.shape[0]) * (
            self.regularization * max(float(np.trace(scatter)), 1.0) / scatter.shape[0]
        )

        try:
            self.weights = np.linalg.solve(scatter, stego_mean - cover_mean)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(scatter, stego_mean - cover_mean, rcond=None)[0]

        self.threshold = float(
            (self.weights @ cover_mean + self.weights @ stego_mean) / 2.0
        )
        return self

    def decide(self, features: np.ndarray) -> np.ndarray:
        return (features @ self.weights >= self.threshold).astype(np.int64)


class EnsembleClassifier:
    """Bagged Fisher discriminants on random feature subspaces."""

    def __init__(
        self,
        base_learners: int = 50,
        subspace_sizes: Sequence[int] = (10, 20, 40, 80),
        seed: int = 20240521,
    ):
        """
        Args:
            base_learners: Number of discriminants in the ensemble.
            subspace_sizes: Candidate feature-subspace widths. The width with
                the lowest out-of-bag error is kept.
            seed: Seed for the bootstrap and subspace draws.
        """
        if base_learners < 1:
            raise ValueError("base_learners must be positive")
        if not subspace_sizes:
            raise ValueError("subspace_sizes must not be empty")

        self.base_learners = base_learners
        self.subspace_sizes = tuple(subspace_sizes)
        self.seed = seed
        self._learners: List[Tuple[np.ndarray, _FisherDiscriminant]] = []
        self.subspace_size: Optional[int] = None
        self.oob_error: Optional[float] = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EnsembleClassifier":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        # A discriminant fitted on nearly as many features as it has examples
        # interpolates the training set and generalises no better than chance,
        # so the subspace is capped at a quarter of the training size.
        ceiling = max(2, min(features.shape[1], len(features) // 4))

        best_error = np.inf
        for size in sorted({min(size, ceiling) for size in self.subspace_sizes}):
            learners, error = self._train(features, labels, size)
            if error < best_error:
                best_error, self._learners, self.subspace_size = error, learners, size

        # An out-of-bag error at or above 0.5 means no subspace found a
        # direction that generalises: read it as "these features carry no
        # evidence against this method", not as a usable detector.
        self.oob_error = float(best_error)
        return self

    def _train(self, features: np.ndarray, labels: np.ndarray, subspace_size: int):
        """Train one ensemble and score it on the out-of-bag samples."""
        rng = np.random.default_rng(self.seed)
        sample_count = len(features)

        learners: List[Tuple[np.ndarray, _FisherDiscriminant]] = []
        votes = np.zeros(sample_count)
        vote_counts = np.zeros(sample_count)

        for _ in range(self.base_learners):
            subspace = rng.choice(features.shape[1], size=subspace_size, replace=False)
            bag = rng.choice(sample_count, size=sample_count, replace=True)

            bag_labels = labels[bag]
            if len(np.unique(bag_labels)) < 2:
                continue

            learner = _FisherDiscriminant().fit(features[np.ix_(bag, subspace)], bag_labels)
            learners.append((subspace, learner))

            # Score on the samples this learner never saw.
            out_of_bag = np.setdiff1d(np.arange(sample_count), bag)
            if out_of_bag.size:
                votes[out_of_bag] += learner.decide(features[np.ix_(out_of_bag, subspace)])
                vote_counts[out_of_bag] += 1

        if not learners:
            raise ValueError("could not train any base learner; check the labels")

        scored = vote_counts > 0
        if not scored.any():
            return learners, 0.5

        predictions = (votes[scored] / vote_counts[scored] >= 0.5).astype(np.int64)
        return learners, float(np.mean(predictions != labels[scored]))

    def decision_scores(self, features: np.ndarray) -> np.ndarray:
        """Fraction of base learners voting 'stego' for each sample."""
        if not self._learners:
            raise ValueError("the classifier has not been fitted")

        features = np.asarray(features, dtype=np.float64)
        votes = np.zeros(len(features))
        for subspace, learner in self._learners:
            votes += learner.decide(features[:, subspace])
        return votes / len(self._learners)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.decision_scores(features) >= 0.5).astype(np.int64)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Detection accuracy; 0.5 means the method is undetectable here."""
        return float(np.mean(self.predict(features) == np.asarray(labels, dtype=np.int64)))
