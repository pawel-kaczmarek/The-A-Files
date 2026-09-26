"""Statistical analysis of experiment results.

The rows of an experiment are not independent observations. Every file
contributes many rows (payload lengths, repetitions, attack variants), and
rows of the same file share its content, so treating them as independent
makes confidence intervals too narrow. The analysis therefore takes the
*file* as the unit of replication:

* uncertainty comes from a cluster bootstrap that resamples whole files;
* methods are compared on per-file means, in a paired design, with
  rank-based tests (Wilcoxon signed-rank for two methods, Friedman with
  Holm-corrected pairwise Wilcoxon tests for more), following Demsar (2006),
  "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 7;
* methods are summarised by their Pareto front instead of a single weighted
  score, whose ranking depends on arbitrary weights and on which other
  methods happen to be in the comparison.

Failed trials are not bit errors. They are counted separately
(``completion_rate``); where a single robustness figure has to include them,
a failed extraction is scored at chance level, BER 0.5 - it recovers no
information, but it is not worse than guessing either.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Hashable, Iterable, Sequence

import numpy as np

#: BER assigned to a trial that produced no decoded message, when failures
#: must be folded into one robustness figure: chance level for binary data.
FAILURE_IMPUTED_BER = 0.5

CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
#: Fixed so that the reported intervals of a result set never change between
#: two summaries of the same rows.
BOOTSTRAP_SEED = 0
ALPHA = 0.05


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[Hashable],
    confidence: float = CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    """Percentile interval of the pooled mean, resampling whole clusters.

    Each resample draws clusters with replacement and pools all their
    values, which keeps the dependence between values of one cluster. With a
    single cluster there is nothing to resample and no interval is given.
    """
    sums: dict[Hashable, float] = {}
    counts: dict[Hashable, int] = {}
    for value, cluster in zip(values, clusters):
        sums[cluster] = sums.get(cluster, 0.0) + float(value)
        counts[cluster] = counts.get(cluster, 0) + 1
    if len(sums) < 2:
        return None, None

    keys = list(sums)
    sum_array = np.array([sums[key] for key in keys])
    count_array = np.array([counts[key] for key in keys])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(resamples, len(keys)))
    means = sum_array[picks].sum(axis=1) / count_array[picks].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def estimate(values: Sequence[float | None], clusters: Sequence[Hashable]) -> dict[str, Any]:
    """Pooled mean of the finite values with a cluster-bootstrap interval."""
    pairs = [(float(v), c) for v, c in zip(values, clusters) if _finite(v)]
    if not pairs:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n": 0, "clusters": 0}
    kept_values = [v for v, _ in pairs]
    kept_clusters = [c for _, c in pairs]
    low, high = cluster_bootstrap_ci(kept_values, kept_clusters)
    return {
        "estimate": sum(kept_values) / len(kept_values),
        "ci95_low": low,
        "ci95_high": high,
        "n": len(kept_values),
        "clusters": len(set(kept_clusters)),
    }


def trial_ber(row: Any, impute_failures: bool = False) -> float | None:
    """BER of one result row; a failed trial is ``None`` or chance level."""
    if row.status == "ok":
        return row.ber
    return FAILURE_IMPUTED_BER if impute_failures else None


def file_of(row: Any) -> str:
    return row.file_name


# --------------------------------------------------------------------------
# Paired comparison of methods
# --------------------------------------------------------------------------


def block_means(
    rows: Iterable[Any],
    value: Callable[[Any], float | None],
    treatment: Callable[[Any], Hashable],
    block: Callable[[Any], Hashable] = file_of,
) -> dict[Hashable, dict[Hashable, float]]:
    """``{treatment: {block: mean value}}`` over finite values."""
    sums: dict[Hashable, dict[Hashable, list[float]]] = {}
    for row in rows:
        observed = value(row)
        if not _finite(observed):
            continue
        sums.setdefault(treatment(row), {}).setdefault(block(row), []).append(float(observed))
    return {
        name: {key: sum(items) / len(items) for key, items in blocks.items()}
        for name, blocks in sums.items()
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for position, index in enumerate(order):
        running = max(running, min(1.0, (total - position) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _wilcoxon(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """(p-value, matched-pairs rank-biserial correlation) of first vs second."""
    from scipy.stats import rankdata, wilcoxon

    differences = first - second
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 1.0, 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    effect = (positive - negative) / (positive + negative)
    try:
        p_value = float(wilcoxon(first, second).pvalue)
    except ValueError:
        p_value = 1.0
    return (p_value if math.isfinite(p_value) else 1.0), effect


def min_blocks_for_significance(treatments: int, alpha: float = ALPHA) -> int:
    """Fewest files at which a Holm-corrected pairwise Wilcoxon test can reach alpha.

    The smallest two-sided exact p-value of the signed-rank test with n pairs
    is 2 / 2**n, and the first Holm step multiplies it by the number of method
    pairs. Below this many files no pairwise difference can be significant,
    whatever the data: 6 files for two methods, 7 for three, 14 for 27.
    """
    pairs = max(1, treatments * (treatments - 1) // 2)
    blocks = 1
    while pairs * 2.0 / 2 ** blocks >= alpha:
        blocks += 1
    return blocks


def paired_comparison(
    rows: Iterable[Any],
    value: Callable[[Any], float | None],
    higher_is_better: bool,
    treatment: Callable[[Any], Hashable] = lambda row: row.method,
    block: Callable[[Any], Hashable] = file_of,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Compare treatments (methods) measured on the same blocks (files).

    Only blocks observed for every treatment are used, so each test is
    paired. Reports mean ranks (1 = best), an omnibus test, Holm-corrected
    pairwise Wilcoxon signed-rank tests with rank-biserial effect sizes, and
    the Nemenyi critical difference for a critical-difference diagram.
    """
    means = block_means(rows, value, treatment, block)
    treatments = sorted(means, key=str)
    if len(treatments) < 2:
        return {"available": False, "reason": "fewer than two methods with results"}

    shared = set.intersection(*(set(means[name]) for name in treatments))
    blocks = sorted(shared, key=str)
    if len(blocks) < 2:
        return {
            "available": False,
            "reason": "fewer than two files observed for every method",
            "treatments": treatments,
        }

    matrix = np.array([[means[name][key] for name in treatments] for key in blocks])
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    oriented = -matrix if higher_is_better else matrix
    ranks = np.vstack([rankdata(row) for row in oriented])
    mean_ranks = {name: float(rank) for name, rank in zip(treatments, ranks.mean(axis=0))}

    count, blocks_n = len(treatments), len(blocks)
    omnibus: dict[str, Any]
    if count == 2:
        p_value, _ = _wilcoxon(matrix[:, 0], matrix[:, 1])
        omnibus = {"test": "wilcoxon_signed_rank", "statistic": None, "p_value": p_value}
    else:
        try:
            with warnings.catch_warnings():
                # All-tied data divide by zero inside scipy; handled below.
                warnings.simplefilter("ignore", RuntimeWarning)
                result = friedmanchisquare(*matrix.T)
            statistic, p_value = float(result.statistic), float(result.pvalue)
        except ValueError:
            statistic, p_value = float("nan"), float("nan")
        if not math.isfinite(p_value):
            # Every method scored identically on every file.
            statistic, p_value = 0.0, 1.0
        omnibus = {"test": "friedman", "statistic": statistic, "p_value": p_value}
    omnibus["significant"] = omnibus["p_value"] < alpha

    pairs: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for first in range(count):
        for second in range(first + 1, count):
            p_value, effect = _wilcoxon(matrix[:, first], matrix[:, second])
            difference = float(np.median(matrix[:, first] - matrix[:, second]))
            pairs.append(
                {
                    "a": treatments[first],
                    "b": treatments[second],
                    "median_difference": difference,
                    # Positive when a scores higher than b on most files.
                    "rank_biserial": effect,
                    "better": (
                        None
                        if difference == 0
                        else treatments[first]
                        if (difference > 0) == higher_is_better
                        else treatments[second]
                    ),
                    "p_value": p_value,
                }
            )
            raw_p.append(p_value)
    for pair, adjusted in zip(pairs, holm_adjust(raw_p)):
        pair["p_holm"] = adjusted
        pair["significant"] = adjusted < alpha

    q_alpha = float(studentized_range.ppf(1.0 - alpha, count, np.inf)) / math.sqrt(2.0)
    critical_difference = q_alpha * math.sqrt(count * (count + 1) / (6.0 * blocks_n))

    return {
        "available": True,
        "higher_is_better": higher_is_better,
        "blocks": blocks_n,
        "treatments": treatments,
        "mean_ranks": dict(sorted(mean_ranks.items(), key=lambda item: item[1])),
        "omnibus": omnibus,
        "pairwise": pairs,
        "critical_difference": critical_difference,
        "alpha": alpha,
    }


# --------------------------------------------------------------------------
# Multi-criteria summary
# --------------------------------------------------------------------------


def pareto_front(
    scores: dict[str, dict[str, float | None]],
    directions: dict[str, bool],
) -> dict[str, Any]:
    """Methods not dominated on any objective.

    ``scores`` maps a method to its objective values and ``directions`` an
    objective to ``True`` when higher is better. Only objectives known for
    every method are used; a method dominates another when it is at least as
    good on all of them and strictly better on one. Unlike a weighted score,
    the result does not depend on weights or on rescaling, and adding a method
    can only remove others from the front, never reorder them.
    """
    methods = sorted(scores)
    objectives = [
        name
        for name in sorted(directions)
        if methods and all(_finite(scores[method].get(name)) for method in methods)
    ]
    excluded = sorted(set(directions) - set(objectives))
    if not methods or not objectives:
        return {"objectives": objectives, "excluded_objectives": excluded, "front": methods, "dominated_by": {}}

    def oriented(method: str, objective: str) -> float:
        raw = float(scores[method][objective])
        return raw if directions[objective] else -raw

    dominated_by: dict[str, list[str]] = {}
    for method in methods:
        for other in methods:
            if other == method:
                continue
            at_least = all(oriented(other, o) >= oriented(method, o) for o in objectives)
            strictly = any(oriented(other, o) > oriented(method, o) for o in objectives)
            if at_least and strictly:
                dominated_by.setdefault(method, []).append(other)

    return {
        "objectives": objectives,
        "excluded_objectives": excluded,
        "front": [method for method in methods if method not in dominated_by],
        "dominated_by": dominated_by,
    }


__all__ = [
    "ALPHA",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONFIDENCE",
    "FAILURE_IMPUTED_BER",
    "block_means",
    "cluster_bootstrap_ci",
    "estimate",
    "file_of",
    "holm_adjust",
    "min_blocks_for_significance",
    "paired_comparison",
    "pareto_front",
    "trial_ber",
]
