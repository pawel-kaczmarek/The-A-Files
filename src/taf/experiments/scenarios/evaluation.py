"""Evaluation block shared by every factorial design.

Each design keeps its own primary analysis (a robustness matrix, a capacity
estimate, a trade-off curve, ...). This module adds, from the same rows and
with the same estimators (``taf.experiments.analysis``), the views every
audio-steganography experiment needs to be read properly:

* per method: the distribution of BER without and under attack (mean,
  median, SD, range, quartiles), recovery rates at explicit BER thresholds;
* per attack: the BER increase it causes, paired by file against the same
  file without attack, and which methods resist it;
* per attack family with several strengths: the degradation curve and a
  test of monotonic trend;
* quality (cover vs stego) kept apart from attack damage (stego vs attacked);
* payload and runtime, including the real-time factor;
* rank correlations with the file as the unit, Holm-corrected;
* a list of facts, each derived from a number above, never a free-text
  conclusion.

Every interval, test and correction is named in ``settings`` so that a
reported figure can be reproduced.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

from taf.attacks.base import Severity
from taf.experiments.analysis import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    FAILURE_IMPUTED_BER,
    PERMUTATION_SEED,
    PERMUTATIONS,
    block_means,
    box_summary,
    estimate,
    file_of,
    holm_adjust,
    min_blocks_for_significance,
    paired_comparison,
    paired_difference,
    stratified_spearman,
    trial_ber,
)
from taf.experiments.results import (
    ExperimentResultRow,
    attacked_rows,
    baseline_rows,
    distribution_stats,
    group_by,
    metric_direction,
)
from taf.experiments.scenarios.common import ber_comparison, metric_comparisons
from taf.experiments.schema import ExperimentConfig

#: Bumped whenever the content of the block changes; stored summaries of an
#: older version are recomputed from their rows.
EVALUATION_VERSION = 1

#: Recovery thresholds: exact recovery, and two BER levels commonly used to
#: speak of a payload recoverable with light error correction.
RECOVERY_THRESHOLDS: dict[str, float] = {"exact": 0.0, "le_1pct": 0.01, "le_5pct": 0.05}

#: A method is reported as stable under an attack when the upper 95% bound of
#: its mean BER under that attack does not exceed this value.
STABLE_BER = 0.01

_SEVERITY_ORDER = [level.value for level in Severity]

Row = ExperimentResultRow


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def _imputed(row: Row) -> float | None:
    return trial_ber(row, impute_failures=True)


def _completed_ber(row: Row) -> float | None:
    return row.ber if row.status == "ok" else None


def _file_means(rows: Sequence[Row], value: Callable[[Row], float | None]) -> dict[Hashable, float]:
    return block_means(rows, value, treatment=lambda row: 0).get(0, {})


def _recovery(rows: Sequence[Row]) -> dict[str, Any]:
    """Share of trials that recovered the payload within each threshold.

    Over all trials: a failed trial did not recover the payload.
    """
    files = [file_of(row) for row in rows]
    return {
        name: estimate(
            [1.0 if row.status == "ok" and row.ber is not None and row.ber <= threshold + 1e-12 else 0.0 for row in rows],
            files,
        )
        for name, threshold in RECOVERY_THRESHOLDS.items()
    }


def _failures(rows: Sequence[Row]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.status != "ok":
            kind = row.failure_kind or "unknown"
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _ber_block(rows: Sequence[Row]) -> dict[str, Any] | None:
    """Distribution, interval and recovery rates of BER over ``rows``."""
    if not rows:
        return None
    completed = [row for row in rows if row.status == "ok" and row.ber is not None]
    return {
        "trials": len(rows),
        "files": len({file_of(row) for row in rows}),
        "completion_rate": len(completed) / len(rows),
        "failures": _failures(rows),
        # Descriptive statistics over completed trials.
        "ber": distribution_stats([row.ber for row in completed], [file_of(row) for row in completed]),
        "box": box_summary([row.ber for row in completed]),
        # Mean with failures at chance level: the figure methods are ranked on.
        "ber_imputed": estimate([_imputed(row) for row in rows], [file_of(row) for row in rows]),
        "recovery": _recovery(rows),
    }


def attack_level(label: str | None, sweep: dict[str, tuple[int, Any]] | None = None) -> dict[str, Any]:
    """Family, level and ordinal strength of an attack label.

    ``"mp3@strong"`` is family ``mp3`` at rank 3 of the named severities; a
    swept specification takes its rank from the sweep order; the clean
    condition is rank 0. Explicit parameter sets without a severity have no
    rank, since their strength relative to other settings is not declared.
    """
    if label is None:
        return {"family": None, "level": "clean", "rank": 0}
    if sweep and label in sweep:
        position, value = sweep[label]
        return {"family": label.split(":", 1)[0].split("@", 1)[0].strip(), "level": str(value), "rank": position + 1}
    name, _, severity = label.partition("@")
    family = name.split(":", 1)[0].strip()
    severity = severity.strip().lower()
    if severity in _SEVERITY_ORDER:
        return {"family": family, "level": severity, "rank": _SEVERITY_ORDER.index(severity) + 1}
    return {"family": family, "level": label, "rank": None}


def _estimate_value(entry: dict[str, Any] | None) -> float | None:
    return None if not entry else entry.get("estimate")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _method_section(rows: Sequence[Row], methods: list[str]) -> list[dict[str, Any]]:
    clean = baseline_rows(rows)
    attacked = attacked_rows(rows)
    section = []
    for method in methods:
        section.append(
            {
                "method": method,
                "clean": _ber_block([row for row in clean if row.method == method]),
                "attacked": _ber_block([row for row in attacked if row.method == method]),
                # Spread between files, the variability a new recording would meet.
                "between_file_sd": _between_file_sd(
                    [row for row in (attacked or clean) if row.method == method]
                ),
            }
        )
    return section


def _between_file_sd(rows: Sequence[Row]) -> float | None:
    means = list(_file_means(rows, _imputed).values())
    if len(means) < 2:
        return None
    mean = sum(means) / len(means)
    return (sum((value - mean) ** 2 for value in means) / (len(means) - 1)) ** 0.5


def _attack_section(
    rows: Sequence[Row], methods: list[str], sweep: dict[str, tuple[int, Any]] | None
) -> list[dict[str, Any]]:
    clean = baseline_rows(rows)
    clean_means = {method: _file_means([r for r in clean if r.method == method], _imputed) for method in methods}
    labels = sorted(
        {row.attack for row in rows if row.attack is not None},
        key=lambda label: (
            attack_level(label, sweep)["family"] or "",
            attack_level(label, sweep)["rank"] if attack_level(label, sweep)["rank"] is not None else 99,
            label,
        ),
    )
    section = []
    for label in labels:
        group = [row for row in rows if row.attack == label]
        level = attack_level(label, sweep)
        per_method = []
        for method in methods:
            own = [row for row in group if row.method == method]
            if not own:
                continue
            block = _ber_block(own)
            per_method.append(
                {
                    "method": method,
                    **block,
                    # BER increase over the same files without the attack.
                    "delta_ber": paired_difference(_file_means(own, _imputed), clean_means[method]),
                }
            )
        resistant = sorted(
            (entry for entry in per_method if _estimate_value(entry["ber_imputed"]) is not None),
            key=lambda entry: entry["ber_imputed"]["estimate"],
        )
        section.append(
            {
                "attack": label,
                **level,
                "parameters": group[0].attack_parameters if group else {},
                "per_method": per_method,
                "most_resistant": [entry["method"] for entry in resistant],
            }
        )
    return section


def _most_destructive(attacks: list[dict[str, Any]], methods: list[str]) -> dict[str, list[dict[str, Any]]]:
    ranking: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        entries = []
        for attack in attacks:
            for entry in attack["per_method"]:
                if entry["method"] == method and _estimate_value(entry["delta_ber"]) is not None:
                    entries.append({"attack": attack["attack"], "delta_ber": entry["delta_ber"]})
        entries.sort(key=lambda item: -item["delta_ber"]["estimate"])
        ranking[method] = entries
    return ranking


def _severity_section(
    rows: Sequence[Row],
    methods: list[str],
    attacks: list[dict[str, Any]],
    sweep: dict[str, tuple[int, Any]] | None,
    parameter: str | None,
) -> list[dict[str, Any]]:
    """Degradation curves for attack families observed at two or more strengths."""
    clean = baseline_rows(rows)
    families: dict[str, list[dict[str, Any]]] = {}
    for attack in attacks:
        if attack["rank"] is not None and attack["family"]:
            families.setdefault(attack["family"], []).append(attack)
    section = []
    for family, levels in sorted(families.items()):
        if len({level["rank"] for level in levels}) < 2:
            continue
        levels = sorted(levels, key=lambda level: level["rank"])
        curves = []
        for method in methods:
            own_clean = [row for row in clean if row.method == method]
            clean_means = _file_means(own_clean, _imputed)
            points = [
                {
                    "level": "clean",
                    "rank": 0,
                    "attack": None,
                    "ber": estimate([_imputed(r) for r in own_clean], [file_of(r) for r in own_clean]),
                    "delta_ber": None,
                }
            ]
            trend_rows = list(own_clean)
            for level in levels:
                own = [row for row in rows if row.attack == level["attack"] and row.method == method]
                trend_rows += own
                points.append(
                    {
                        "level": level["level"],
                        "rank": level["rank"],
                        "attack": level["attack"],
                        "ber": estimate([_imputed(r) for r in own], [file_of(r) for r in own]),
                        "delta_ber": paired_difference(_file_means(own, _imputed), clean_means),
                    }
                )
            ranks = {level["attack"]: level["rank"] for level in levels}
            trend = stratified_spearman(
                [0 if row.attack is None else ranks[row.attack] for row in trend_rows],
                [_imputed(row) for row in trend_rows],
                [file_of(row) for row in trend_rows],
            )
            curves.append({"method": method, "points": points, "trend": trend})
        section.append(
            {
                "family": family,
                "parameter": parameter if sweep else "severity",
                "levels": ["clean"] + [level["level"] for level in levels],
                "curves": curves,
            }
        )
    return section


def _quality_section(rows: Sequence[Row], methods: list[str]) -> dict[str, Any]:
    clean = baseline_rows(rows)
    attacked = attacked_rows(rows)
    names = sorted({name for row in clean for name in row.metrics})
    metrics = [
        {
            "name": name,
            "higher_is_better": metric_direction(name),
            "per_method": [
                {
                    "method": method,
                    **distribution_stats(
                        [row.metrics.get(name) for row in clean if row.method == method],
                        [file_of(row) for row in clean if row.method == method],
                    ),
                }
                for method in methods
            ],
        }
        for name in names
    ]
    damage_names = sorted({name for row in attacked for name in row.attack_metrics})
    damage = []
    for name in damage_names:
        per_attack = []
        for attack, group in sorted(group_by(attacked, lambda row: row.attack).items()):
            per_attack.append(
                {
                    "attack": attack,
                    "per_method": [
                        {
                            "method": method,
                            **distribution_stats(
                                [row.attack_metrics.get(name) for row in group if row.method == method],
                                [file_of(row) for row in group if row.method == method],
                            ),
                        }
                        for method in methods
                        if any(row.method == method for row in group)
                    ],
                }
            )
        damage.append({"name": name, "higher_is_better": metric_direction(name), "per_attack": per_attack})
    all_names = set(names) | set(damage_names)
    return {
        # Cover vs stego, measured once per embedded signal, before any attack.
        "metrics": metrics,
        # Stego vs attacked: what the attack did to the signal, not the embedding.
        "attack_damage": damage,
        "psnr_available": any("PSNR" in name.upper() for name in all_names),
    }


def _payload_section(rows: Sequence[Row], methods: list[str]) -> dict[str, Any]:
    clean = baseline_rows(rows)
    levels = sorted({row.payload_length for row in clean})
    names = sorted({name for row in clean for name in row.metrics})
    per_method = []
    for method in methods:
        points = []
        for payload in levels:
            own = [row for row in clean if row.method == method and row.payload_length == payload]
            if not own:
                continue
            files = [file_of(row) for row in own]
            points.append(
                {
                    "payload_bits": payload,
                    "payload_bps": distribution_stats([row.payload_rate_bps for row in own], files),
                    # Completed trials: over-capacity refusals are in completion_rate.
                    "ber": estimate([_completed_ber(row) for row in own], files),
                    "ber_imputed": estimate([_imputed(row) for row in own], files),
                    "completion_rate": sum(1 for row in own if row.status == "ok") / len(own),
                    "quality": {name: estimate([row.metrics.get(name) for row in own], files) for name in names},
                }
            )
        per_method.append({"method": method, "points": points})
    return {"levels": levels, "available": len(levels) >= 2, "metrics": names, "per_method": per_method}


def _total_time(row: Row) -> float | None:
    if row.encode_time_seconds is None or row.decode_time_seconds is None:
        return None
    return row.encode_time_seconds + row.decode_time_seconds


def _real_time_factor(row: Row) -> float | None:
    total = _total_time(row)
    if total is None or not row.duration_seconds:
        return None
    return total / row.duration_seconds


def _runtime_section(rows: Sequence[Row], methods: list[str], config: ExperimentConfig) -> dict[str, Any]:
    # Clean rows only: one encode and one decode per embedded signal. Attacked
    # rows reuse the same encoding, and the attack is not part of the method.
    clean = baseline_rows(rows)
    workers = getattr(config, "max_workers", None)
    per_method = []
    for method in methods:
        own = [row for row in clean if row.method == method and row.status == "ok"]
        files = [file_of(row) for row in own]
        per_method.append(
            {
                "method": method,
                "encode_seconds": distribution_stats([row.encode_time_seconds for row in own], files),
                "decode_seconds": distribution_stats([row.decode_time_seconds for row in own], files),
                "total_seconds": distribution_stats([_total_time(row) for row in own], files),
                "real_time_factor": distribution_stats([_real_time_factor(row) for row in own], files),
            }
        )
    return {
        "max_workers": workers,
        # With several workers, trials compete for the CPU and times are not
        # comparable between methods.
        "comparable": workers == 1,
        "per_method": per_method,
    }


def _correlations(
    rows: Sequence[Row],
    methods: list[str],
    severity: list[dict[str, Any]],
    payload: dict[str, Any],
    timing_comparable: bool,
) -> list[dict[str, Any]]:
    clean = baseline_rows(rows)
    tests: list[dict[str, Any]] = []
    if payload["available"]:
        for method in methods:
            # Completed trials: a payload the method refuses (over capacity) is
            # a capacity limit, reported as completion rate, not a bit error.
            completed = [row for row in clean if row.method == method and row.status == "ok"]
            files = [file_of(row) for row in completed]
            x = [row.payload_length for row in completed]
            tests.append({"x": "payload_bits", "y": "ber", "method": method, "group": None,
                          **stratified_spearman(x, [row.ber for row in completed], files)})
            for name in payload["metrics"]:
                tests.append({"x": "payload_bits", "y": f"metric: {name}", "method": method, "group": None,
                              **stratified_spearman(x, [row.metrics.get(name) for row in completed], files)})
            if timing_comparable:
                timing = stratified_spearman(x, [_total_time(row) for row in completed], files)
            else:
                timing = {"available": False, "reason": "timings were measured with parallel workers"}
            tests.append({"x": "payload_bits", "y": "total_seconds", "method": method, "group": None, **timing})
    for family in severity:
        for curve in family["curves"]:
            # The curve's own trend object, so the Holm-adjusted p-value added
            # below is also what the degradation curve reports.
            trend = curve["trend"]
            trend.update({"x": "attack_strength", "y": "ber", "method": curve["method"], "group": family["family"]})
            tests.append(trend)
    available = [test for test in tests if test.get("available")]
    for test, adjusted in zip(available, holm_adjust([test["p_value"] for test in available])):
        test["p_holm"] = adjusted
        test["significant"] = adjusted < ALPHA
    return tests


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "< 0.001" if value < 0.001 else f"= {value:.3f}"


def _facts(evaluation: dict[str, Any], statistics: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []

    # Largest BER increase per method, reported only when its interval excludes 0.
    for method, entries in evaluation["most_destructive"].items():
        if not entries:
            continue
        top = entries[0]
        delta = top["delta_ber"]
        if delta["ci95_low"] is not None and delta["ci95_low"] > 0:
            facts.append({
                "kind": "largest_ber_increase", "method": method, "attack": top["attack"],
                "delta": delta["estimate"], "ci95_low": delta["ci95_low"], "ci95_high": delta["ci95_high"],
                "files": delta["clusters"],
                "text": f"For {method}, {top['attack']} caused the largest BER increase over the clean condition: "
                        f"+{_fmt(delta['estimate'])} (95% CI {_fmt(delta['ci95_low'])} to {_fmt(delta['ci95_high'])}, "
                        f"{delta['clusters']} files).",
            })
        elif delta["ci95_high"] is not None:
            facts.append({
                "kind": "no_confirmed_increase", "method": method, "attack": top["attack"],
                "delta": delta["estimate"], "ci95_low": delta["ci95_low"], "ci95_high": delta["ci95_high"],
                "text": f"For {method}, no attack raised BER with a 95% interval excluding zero; the largest mean "
                        f"increase was {_fmt(delta['estimate'])} under {top['attack']}.",
            })

    # Methods stable under an attack.
    for attack in evaluation["attacks"]:
        stable = [
            entry for entry in attack["per_method"]
            if entry["ber_imputed"]["ci95_high"] is not None and entry["ber_imputed"]["ci95_high"] <= STABLE_BER
        ]
        if stable:
            facts.append({
                "kind": "stable_under_attack", "attack": attack["attack"],
                "methods": [entry["method"] for entry in stable],
                "bounds": [entry["ber_imputed"]["ci95_high"] for entry in stable],
                "threshold": STABLE_BER,
                "text": f"Under {attack['attack']}, the upper 95% bound of mean BER stayed at or below {STABLE_BER} for "
                        + ", ".join(f"{entry['method']} ({_fmt(entry['ber_imputed']['ci95_high'])})" for entry in stable)
                        + ".",
            })

    # Failures.
    for entry in evaluation["methods"]:
        failed: dict[str, int] = {}
        trials = 0
        for block in (entry["clean"], entry["attacked"]):
            if block:
                trials += block["trials"]
                for kind, count in block["failures"].items():
                    failed[kind] = failed.get(kind, 0) + count
        if failed:
            total = sum(failed.values())
            facts.append({
                "kind": "failures", "method": entry["method"], "failed": total, "trials": trials, "by_kind": failed,
                "text": f"{entry['method']} produced no decoded message in {total} of {trials} trials ("
                        + ", ".join(f"{kind}: {count}" for kind, count in sorted(failed.items())) + ").",
            })

    # Between-file variability.
    spreads = [(entry["method"], entry["between_file_sd"]) for entry in evaluation["methods"] if entry["between_file_sd"] is not None]
    if len(spreads) >= 2:
        spreads.sort(key=lambda item: item[1])
        (low_method, low), (high_method, high) = spreads[0], spreads[-1]
        if high > low:
            facts.append({
                "kind": "variability", "highest": high_method, "highest_sd": high, "lowest": low_method, "lowest_sd": low,
                "text": f"Between-file variability of BER (SD of per-file means) was largest for {high_method} "
                        f"({_fmt(high)}) and smallest for {low_method} ({_fmt(low)}).",
            })

    # Correlations: trends with attack strength and payload.
    tested = [test for test in evaluation["correlations"] if test.get("available")]
    for test in tested:
        if not test["significant"]:
            continue
        direction = "increased" if test["rho"] > 0 else "decreased"
        if test["x"] == "attack_strength":
            subject = f"BER of {test['method']} {direction} with the strength of {test['group']}"
        elif test["y"] == "ber":
            subject = f"BER of {test['method']} {direction} with payload length"
        elif test["y"] == "total_seconds":
            subject = f"Processing time of {test['method']} {direction} with payload length"
        else:
            subject = f"{test['y'].removeprefix('metric: ')} of {test['method']} {direction} with payload length"
        facts.append({
            "kind": "correlation", **{key: test[key] for key in ("x", "y", "method", "group", "rho", "p_holm", "n_files")},
            "text": f"{subject} (Spearman ρ = {_fmt(test['rho'], 2)}, Holm-adjusted p {_p(test['p_holm'])}, "
                    f"{test['n_files']} files).",
        })
    if tested:
        significant = sum(1 for test in tested if test["significant"])
        facts.append({
            "kind": "correlations_tested", "significant": significant, "tested": len(tested),
            "text": f"{significant} of {len(tested)} rank-correlation tests were significant after Holm correction "
                    f"(α = {ALPHA}).",
        })

    # Differences between methods.
    for key, comparison in statistics.items():
        if not isinstance(comparison, dict) or "available" not in comparison or not comparison.get("available"):
            continue
        omnibus = comparison["omnibus"]
        pairs = comparison.get("pairwise", [])
        significant_pairs = [pair for pair in pairs if pair.get("significant")]
        # Among significant pairs only: the rank-biserial correlation ignores
        # tied files, so a non-significant pair can show |r| = 1 from one file.
        largest = max(significant_pairs, key=lambda pair: abs(pair["rank_biserial"]), default=None)
        needed = min_blocks_for_significance(len(comparison["treatments"]))
        test = "Friedman" if omnibus["test"] == "friedman" else "Wilcoxon signed-rank"
        outcome = "detected" if omnibus["significant"] else "did not detect"
        text = (
            f"On {key.replace('_', ' ')}, the {test} test {outcome} a difference among "
            f"{len(comparison['treatments'])} methods (p {_p(omnibus['p_value'])}, {comparison['blocks']} files); "
            f"{len(significant_pairs)} of {len(pairs)} pairwise comparisons were significant after Holm correction."
        )
        effect = None
        if largest is not None:
            # "better" is None when the median difference is zero.
            better = largest["better"] or largest["a"]
            effect = {
                "better": better,
                "worse": largest["b"] if better == largest["a"] else largest["a"],
                "directional": largest["better"] is not None,
                "r": abs(largest["rank_biserial"]),
            }
            relation = "better than" if effect["directional"] else "vs"
            text += (
                f" Largest effect: {effect['better']} {relation} {effect['worse']}, "
                f"rank-biserial |r| = {_fmt(effect['r'], 2)}."
            )
        if comparison["blocks"] < needed:
            text += f" With {comparison['blocks']} files no Holm-corrected pairwise test can reach α = {ALPHA}; {needed} are needed."
        facts.append({
            "kind": "method_difference", "comparison": key, "test": omnibus["test"], "p_value": omnibus["p_value"],
            "significant": omnibus["significant"], "files": comparison["blocks"], "methods": len(comparison["treatments"]),
            "pairs_significant": len(significant_pairs), "pairs": len(pairs),
            "largest_effect": effect,
            "underpowered": comparison["blocks"] < needed, "min_files": needed, "text": text,
        })
    return facts


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def evaluation_statistics(rows: Sequence[Row], config: ExperimentConfig) -> dict[str, Any]:
    """Paired comparisons the evaluation adds to ``summary["statistics"]``."""
    clean = baseline_rows(rows)
    added: dict[str, Any] = {}
    several = len({row.method for row in rows}) >= 2
    attacks = sorted({row.attack for row in rows if row.attack is not None})
    if attacks and several:
        added["per_attack"] = {attack: ber_comparison([row for row in rows if row.attack == attack]) for attack in attacks}
    metrics = metric_comparisons(clean)
    if metrics:
        added["metrics"] = metrics
    # Processing time is compared only when it was measured one trial at a time.
    if several and getattr(config, "max_workers", None) == 1:
        added["runtime_total"] = paired_comparison(
            clean, lambda row: _total_time(row) if row.status == "ok" else None, higher_is_better=False
        )
    return added


def evaluation_summary(
    rows: Sequence[Row], config: ExperimentConfig, statistics: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The shared evaluation block for a completed factorial experiment."""
    methods = sorted({row.method for row in rows})
    sweep_config = getattr(config, "attack_sweep", None)
    sweep: dict[str, tuple[int, Any]] | None = None
    if sweep_config is not None:
        from taf.experiments.sweeps import attack_sweep_specs

        sweep = {spec: (position, value) for position, (spec, value) in enumerate(attack_sweep_specs(sweep_config))}

    attacks = _attack_section(rows, methods, sweep)
    severity = _severity_section(rows, methods, attacks, sweep, sweep_config.parameter if sweep_config else None)
    payload = _payload_section(rows, methods)
    runtime = _runtime_section(rows, methods, config)
    payload_lengths = sorted({row.payload_length for row in rows if row.payload_length})
    smallest = payload_lengths[0] if payload_lengths else None

    if statistics and "runtime_total" in statistics:
        runtime["comparison"] = statistics["runtime_total"]

    evaluation: dict[str, Any] = {
        "version": EVALUATION_VERSION,
        "settings": {
            "unit_of_replication": "file",
            "aggregation": "mean over trials; per-file means for paired tests and correlations",
            "descriptive_statistics": "completed trials; quartiles by linear interpolation (Hyndman-Fan type 7)",
            "confidence": CONFIDENCE,
            "interval": "percentile cluster bootstrap over files",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "failure_policy": f"failed trials excluded from descriptive BER, scored BER = {FAILURE_IMPUTED_BER} in means "
                              "used for ranking and BER increase, and counted as not recovered in recovery rates",
            "recovery_thresholds": RECOVERY_THRESHOLDS,
            "stable_ber": STABLE_BER,
            "paired_test_two": "Wilcoxon signed-rank",
            "paired_test_many": "Friedman, then pairwise Wilcoxon signed-rank",
            "correction": "Holm",
            "effect_size": "matched-pairs rank-biserial correlation",
            "correlation": "Spearman on per-(file, level) means",
            "correlation_test": "permutation within files",
            "permutations": PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "correlation_correction": "Holm over all correlation tests of the run",
            "alpha": ALPHA,
        },
        "resolution": {
            "min_payload_bits": smallest,
            # With L bits BER moves in steps of 1/L: below 100 bits, BER <= 1%
            # can only mean BER = 0.
            "ber_step": 1.0 / smallest if smallest else None,
            "le_1pct_equals_exact": bool(smallest and smallest < 100),
        },
        "methods": _method_section(rows, methods),
        "attacks": attacks,
        "most_destructive": _most_destructive(attacks, methods),
        "severity": severity,
        "quality": _quality_section(rows, methods),
        "payload": payload,
        "runtime": runtime,
        "correlations": _correlations(rows, methods, severity, payload, runtime["comparable"]),
    }
    evaluation["facts"] = _facts(evaluation, statistics or {})
    return evaluation


__all__ = [
    "EVALUATION_VERSION",
    "RECOVERY_THRESHOLDS",
    "STABLE_BER",
    "attack_level",
    "evaluation_statistics",
    "evaluation_summary",
]
