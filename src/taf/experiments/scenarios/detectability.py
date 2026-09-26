"""Detectability: steganalysis of each method on the dataset."""

from __future__ import annotations

from taf.experiments.scenarios.base import Scenario
from taf.experiments.schema import ExperimentConfig, ExperimentType


def _validate(config: ExperimentConfig) -> list[str]:
    problems: list[str] = []
    if config.attacks or config.attack_preset:
        problems.append("Detectability does not apply attacks; remove them.")
    return problems


SCENARIO = Scenario(
    experiment_type=ExperimentType.DETECTABILITY,
    title="Detectability",
    description=(
        "Train a steganalyser to separate covers from stego signals of each method and "
        "report its held-out accuracy with a confidence interval and a test against chance."
    ),
    property="security",
    factors=("method", "payload_length"),
    measures=("detector_accuracy", "false_positive_rate", "false_negative_rate"),
    analyses=("ensemble_steganalysis", "wilson_ci", "binomial_test"),
    validate=_validate,
    # Results come from taf.experiments.detectability, not from trial rows.
    summarize=lambda rows, config: {},
)
