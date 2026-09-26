"""Robustness curve: bit error rate as a function of attack strength.

The dose-response design of the watermarking literature: one attack, one of
its parameters swept from the mildest to the harshest setting, and every
method measured at every setting on the same files and with the same noise
realisations. Each method gets a curve with cluster-bootstrap intervals and a
breakdown point - where, in sweep order, its BER first exceeds the usable
threshold. Methods are compared on their per-file mean BER over the whole
sweep, the discrete analogue of the area under the curve.
"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    USABLE_BER_THRESHOLD,
    ExperimentResultRow,
    baseline_rows,
    by_method_attack,
    group_by,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.common import ber_comparison, ranking_from
from taf.experiments.schema import ExperimentConfig, ExperimentType
from taf.experiments.sweeps import attack_sweep_specs, threshold_crossing


def _point(value: Any, spec: str, rows: Sequence[ExperimentResultRow]) -> dict[str, Any]:
    stats = group_stats(rows)
    return {
        "value": value,
        "attack": spec,
        "rows": stats["rows"],
        "ber": stats["ber_imputed"],
        "ber_completed": stats["avg_ber"],
        "completion_rate": stats["completion_rate"],
        "decode_success": stats["decode_success"],
    }


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    specs = attack_sweep_specs(config.attack_sweep)
    swept = {spec for spec, _ in specs}
    sweep_rows = [row for row in rows if row.attack in swept]

    curves: list[dict[str, Any]] = []
    for method, group in sorted(group_by(rows, lambda r: r.method).items()):
        points = [
            _point(value, spec, [row for row in group if row.attack == spec]) for spec, value in specs
        ]
        method_sweep_rows = [row for row in sweep_rows if row.method == method]
        curves.append(
            {
                "method": method,
                "baseline": group_stats(baseline_rows(group))["ber_imputed"],
                "points": points,
                "breakdown": threshold_crossing(
                    [(point["value"], point["ber"]["estimate"]) for point in points],
                    USABLE_BER_THRESHOLD,
                ),
                # Mean over the whole sweep: the discrete area under the curve.
                "mean_ber": group_stats(method_sweep_rows)["ber_imputed"],
            }
        )

    fallback = [
        curve["method"]
        for curve in sorted(
            curves,
            key=lambda c: (c["mean_ber"]["estimate"] is None, c["mean_ber"]["estimate"] or 0.0),
        )
    ]
    comparison = ber_comparison(sweep_rows)
    ranking = ranking_from(comparison, fallback)
    sweep = config.attack_sweep
    return {
        "overall": group_stats(rows),
        "sweep": {
            "kind": "attack",
            "target": sweep.target if sweep else None,
            "parameter": sweep.parameter if sweep else None,
            "values": list(sweep.values) if sweep else [],
            "usable_ber_threshold": USABLE_BER_THRESHOLD,
        },
        "curves": curves,
        "matrix": by_method_attack(rows),
        "robustness_ranking": [
            {
                "method": curve["method"],
                "mean_ber": curve["mean_ber"]["estimate"],
                "mean_ber_ci95_low": curve["mean_ber"]["ci95_low"],
                "mean_ber_ci95_high": curve["mean_ber"]["ci95_high"],
                "breakdown_status": curve["breakdown"]["status"],
                "breakdown_value": curve["breakdown"]["value"],
            }
            for curve in sorted(curves, key=lambda c: ranking.index(c["method"]) if c["method"] in ranking else len(ranking))
        ],
        "most_robust_method": ranking[0] if ranking else None,
        "statistics": {
            "ber_sweep": comparison,
            "per_value": {
                str(value): ber_comparison([row for row in rows if row.attack == spec])
                for spec, value in specs
            },
        },
    }


def _validate(config: ExperimentConfig) -> list[str]:
    if config.attack_sweep is None:
        return ["A robustness curve needs an attack sweep (attack, parameter and ordered values)."]
    return []


SCENARIO = Scenario(
    experiment_type=ExperimentType.ROBUSTNESS_CURVE,
    title="Robustness Curve",
    description=(
        "Sweep one attack parameter from mild to harsh and measure BER at every setting: "
        "a curve per method with confidence bands and its breakdown point."
    ),
    property="robustness",
    factors=("method", "attack_parameter"),
    measures=("ber", "decode_success", "completion_rate"),
    analyses=("cluster_bootstrap_ci", "breakdown_point", "friedman_holm_wilcoxon"),
    requires_sweep="attack",
    validate=_validate,
    summarize=_summarize,
)
