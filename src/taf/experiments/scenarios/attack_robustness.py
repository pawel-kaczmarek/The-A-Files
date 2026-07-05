"""Attack robustness: which method survives attacks best?"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.results import (
    ExperimentResultRow,
    by_method_attack,
    group_by,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.schema import ExperimentConfig, ExperimentType


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    cells = by_method_attack(rows)
    attacked = [row for row in rows if row.attack is not None]

    # Robustness per method = average bit accuracy over attacked rows only.
    robustness: list[dict[str, Any]] = []
    for method, group in sorted(group_by(attacked, lambda r: r.method).items()):
        stats = group_stats(group)
        robustness.append({"method": method, **stats})
    robustness.sort(
        key=lambda entry: (entry["avg_bit_accuracy"] is None, -(entry["avg_bit_accuracy"] or 0.0))
    )

    # Best/worst method per attack, and the most damaging attack overall.
    per_attack: list[dict[str, Any]] = []
    for attack, group in sorted(group_by(attacked, lambda r: r.attack).items()):
        method_stats = [
            {"method": method, **group_stats(method_group)}
            for method, method_group in sorted(group_by(group, lambda r: r.method).items())
        ]
        scored = [entry for entry in method_stats if entry["avg_bit_accuracy"] is not None]
        per_attack.append(
            {
                "attack": attack,
                **group_stats(group),
                "best_method": max(scored, key=lambda e: e["avg_bit_accuracy"])["method"] if scored else None,
                "worst_method": min(scored, key=lambda e: e["avg_bit_accuracy"])["method"] if scored else None,
            }
        )
    damaging = [entry for entry in per_attack if entry["avg_bit_accuracy"] is not None]
    worst_attack = min(damaging, key=lambda e: e["avg_bit_accuracy"])["attack"] if damaging else None

    return {
        "overall": group_stats(rows),
        "attacked_overall": group_stats(attacked),
        "matrix": cells,
        "robustness_ranking": robustness,
        "most_robust_method": robustness[0]["method"] if robustness else None,
        "per_attack": per_attack,
        "worst_attack": worst_attack,
    }


SCENARIO = Scenario(
    experiment_type=ExperimentType.ATTACK_ROBUSTNESS,
    title="Attack Robustness",
    description=(
        "Measure how well each method survives signal attacks: every attack is decoded "
        "next to a no-attack baseline and scored by BER, bit accuracy and decode success."
    ),
    requires_attacks=True,
    summarize=_summarize,
)
