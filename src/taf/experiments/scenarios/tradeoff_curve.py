"""Trade-off curve: imperceptibility against robustness as a method is tuned.

Every information-hiding method trades transparency for robustness through
its embedding strength. This design sweeps one constructor parameter of one
method and measures, at every setting, the cover-vs-stego quality metrics,
the BER without attack and - when attacks are selected - the BER under
attack. Each setting is one point of the curve, with cluster-bootstrap
intervals; the Pareto front says which settings are worth using, and a
Friedman test per measure says whether the parameter has an effect at all.
Methods listed in ``methods`` are measured alongside as fixed reference
points.
"""

from __future__ import annotations

from typing import Any, Sequence

from taf.experiments.analysis import estimate, file_of, paired_comparison, pareto_front, trial_ber
from taf.experiments.results import (
    ExperimentResultRow,
    attacked_rows,
    baseline_rows,
    group_stats,
)
from taf.experiments.scenarios.base import Scenario
from taf.experiments.scenarios.common import ranked_metrics
from taf.experiments.schema import ExperimentConfig, ExperimentType
from taf.experiments.sweeps import method_sweep_specs


def _setting_rows(
    rows: Sequence[ExperimentResultRow], name: str, parameters: dict[str, Any]
) -> list[ExperimentResultRow]:
    return [row for row in rows if row.method_type == name and row.method_parameters == parameters]


def _measure(rows: Sequence[ExperimentResultRow], metrics: dict[str, bool]) -> dict[str, Any]:
    baseline = baseline_rows(rows)
    attacked = attacked_rows(rows)
    clusters = [file_of(row) for row in baseline]
    point: dict[str, Any] = {
        "label": rows[0].method if rows else None,
        "rows": len(rows),
        "ber_baseline": estimate([trial_ber(r, impute_failures=True) for r in baseline], clusters),
        "completion_rate": group_stats(rows)["completion_rate"] if rows else None,
        "metrics": {
            name: estimate([row.metrics.get(name) for row in baseline], clusters) for name in metrics
        },
    }
    if attacked:
        point["ber_attacked"] = estimate(
            [trial_ber(r, impute_failures=True) for r in attacked], [file_of(r) for r in attacked]
        )
    return point


def _summarize(rows: Sequence[ExperimentResultRow], config: ExperimentConfig) -> dict[str, Any]:
    from taf.plugins import parse_method_spec

    sweep = config.method_sweep
    specs = method_sweep_specs(sweep)
    metrics = ranked_metrics(baseline_rows(rows))

    points: list[dict[str, Any]] = []
    sweep_labels: set[str] = set()
    for spec, value in specs:
        name, parameters = parse_method_spec(spec)
        setting = _setting_rows(rows, name, parameters)
        sweep_labels.update(row.method for row in setting)
        points.append({"value": value, "spec": spec, **_measure(setting, metrics)})

    references = []
    for spec in config.methods:
        name, parameters = parse_method_spec(spec)
        setting = _setting_rows(rows, name, parameters)
        if setting and setting[0].method not in sweep_labels:
            references.append({"spec": spec, **_measure(setting, metrics)})

    objectives = {"bit_accuracy": True, **{f"metric: {m}": d for m, d in metrics.items()}}
    scores: dict[str, dict[str, float | None]] = {}
    for point in points:
        ber = point["ber_baseline"]["estimate"]
        entry = {"bit_accuracy": None if ber is None else 1.0 - ber}
        entry.update({f"metric: {m}": point["metrics"][m]["estimate"] for m in metrics})
        if "ber_attacked" in point:
            attacked = point["ber_attacked"]["estimate"]
            entry["robustness"] = None if attacked is None else 1.0 - attacked
            objectives["robustness"] = True
        scores[str(point["value"])] = entry

    sweep_rows = [row for row in rows if row.method in sweep_labels]
    value_of = {point["label"]: str(point["value"]) for point in points if point["label"]}

    def setting(row: ExperimentResultRow) -> str:
        return value_of.get(row.method, row.method)

    statistics: dict[str, Any] = {
        "ber_baseline": paired_comparison(
            baseline_rows(sweep_rows),
            lambda r: trial_ber(r, impute_failures=True),
            higher_is_better=False,
            treatment=setting,
        ),
        "metrics": {
            name: paired_comparison(
                baseline_rows(sweep_rows), lambda r, n=name: r.metrics.get(n), direction, treatment=setting
            )
            for name, direction in metrics.items()
        },
    }
    if attacked_rows(sweep_rows):
        statistics["ber_attacked"] = paired_comparison(
            attacked_rows(sweep_rows),
            lambda r: trial_ber(r, impute_failures=True),
            higher_is_better=False,
            treatment=setting,
        )

    return {
        "overall": group_stats(rows),
        "sweep": {
            "kind": "method",
            "target": sweep.target if sweep else None,
            "parameter": sweep.parameter if sweep else None,
            "values": list(sweep.values) if sweep else [],
        },
        "points": points,
        "references": references,
        "pareto": pareto_front(scores, objectives),
        "statistics": statistics,
        "ranked_metrics": metrics,
        "by_setting": [
            {
                "value": point["value"],
                "method": point["label"],
                "ber_baseline": point["ber_baseline"]["estimate"],
                "ber_attacked": (point.get("ber_attacked") or {}).get("estimate"),
                "completion_rate": point["completion_rate"],
            }
            for point in points
        ],
    }


def _validate(config: ExperimentConfig) -> list[str]:
    if config.method_sweep is None:
        return ["A trade-off curve needs a method sweep (method, parameter and ordered values)."]
    return []


SCENARIO = Scenario(
    experiment_type=ExperimentType.TRADEOFF_CURVE,
    title="Trade-off Curve",
    description=(
        "Sweep a method's embedding-strength parameter and measure quality and robustness "
        "at every setting: the imperceptibility-robustness curve with its Pareto front."
    ),
    property="multi_criteria",
    factors=("method_parameter", "attack"),
    measures=("quality_metrics", "ber", "ber_under_attack"),
    analyses=("cluster_bootstrap_ci", "pareto_front", "friedman_holm_wilcoxon"),
    requires_metrics=True,
    requires_sweep="method",
    validate=_validate,
    summarize=_summarize,
)
