"""Attack robustness: which method survives attacks best?"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    attacked_rows,
    baseline_rows,
    by_method_attack,
    group_by,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.common import ber_comparison, method_pareto, ranking_from
from taf.experiments.schema import ExperimentConfig, ExperimentType


def _accuracy_key(entry: dict[str, Any]) -> tuple[bool, float]:
    value = entry["avg_bit_accuracy_imputed"]
    return value is None, -(value or 0.0)


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    cells = by_method_attack(rows)
    attacked = attacked_rows(rows)

    # Robustness per method over attacked rows only. Failed extractions count
    # at chance level, so a method cannot look robust by crashing whenever an
    # attack makes decoding hard.
    robustness: list[dict[str, Any]] = [
        {"method": method, **group_stats(group)}
        for method, group in sorted(group_by(attacked, lambda r: r.method).items())
    ]
    robustness.sort(key=_accuracy_key)

    # Best/worst method per attack, and the most damaging attack overall.
    per_attack: list[dict[str, Any]] = []
    per_attack_tests: dict[str, Any] = {}
    for attack, group in sorted(group_by(attacked, lambda r: r.attack).items()):
        method_stats = [
            {"method": method, **group_stats(method_group)}
            for method, method_group in sorted(group_by(group, lambda r: r.method).items())
        ]
        scored = [entry for entry in method_stats if entry["avg_bit_accuracy_imputed"] is not None]
        per_attack.append(
            {
                "attack": attack,
                **group_stats(group),
                "best_method": min(scored, key=_accuracy_key)["method"] if scored else None,
                "worst_method": max(scored, key=_accuracy_key)["method"] if scored else None,
            }
        )
        per_attack_tests[attack] = ber_comparison(group)
    damaging = [entry for entry in per_attack if entry["avg_bit_accuracy_imputed"] is not None]
    worst_attack = (
        min(damaging, key=lambda e: e["avg_bit_accuracy_imputed"])["attack"] if damaging else None
    )

    comparison = ber_comparison(attacked)
    ranking = ranking_from(comparison, [entry["method"] for entry in robustness])
    return {
        "overall": group_stats(rows),
        "baseline": group_stats(baseline_rows(rows)),
        "attacked_overall": group_stats(attacked),
        "matrix": cells,
        "robustness_ranking": robustness,
        "most_robust_method": ranking[0] if ranking else None,
        "per_attack": per_attack,
        "worst_attack": worst_attack,
        "statistics": {"ber_attacked": comparison, "per_attack": per_attack_tests},
        "pareto": method_pareto(rows),
    }


SCENARIO = Scenario(
    experiment_type=ExperimentType.ATTACK_ROBUSTNESS,
    title="Attack Robustness",
    description=(
        "Measure how well each method survives signal attacks: every attack is decoded "
        "next to a no-attack baseline and scored by BER, bit accuracy and decode success."
    ),
    property="robustness",
    factors=("method", "attack"),
    measures=("ber", "decode_success", "completion_rate"),
    analyses=("cluster_bootstrap_ci", "robustness_matrix", "friedman_holm_wilcoxon", "pareto_front"),
    requires_attacks=True,
    summarize=_summarize,
)
