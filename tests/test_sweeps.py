"""Parameter sweeps: robustness curves and trade-off curves."""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from pydantic import ValidationError

from taf.experiments import ExperimentConfig, ExperimentType, run_experiment
from taf.experiments.sweeps import (
    ParameterSweep,
    attack_sweep_specs,
    method_sweep_specs,
    threshold_crossing,
)


def _config(**overrides) -> ExperimentConfig:
    base = dict(
        experiment_type=ExperimentType.ROBUSTNESS_CURVE,
        name="sweep",
        dataset_id="vctk",
        file_limit=2,
        methods=["QIM_METHOD"],
        payload_lengths=[8],
        random_seed=2,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_sweeps_expand_into_specifications_in_order():
    attack = ParameterSweep(target="mp3:align_delay=True", parameter="bitrate_kbps", values=[128, 64, 32])
    assert [value for _, value in attack_sweep_specs(attack)] == [128, 64, 32]
    assert attack_sweep_specs(attack)[0][0] == "mp3:align_delay=True,bitrate_kbps=128"

    method = ParameterSweep(target="QIM_METHOD", parameter="step_scale", values=[0.05, 0.2])
    assert [spec for spec, _ in method_sweep_specs(method)] == [
        "QIM_METHOD:step_scale=0.05",
        "QIM_METHOD:step_scale=0.2",
    ]
    config = _config(experiment_type=ExperimentType.TRADEOFF_CURVE, metrics=["SNR_METRIC"], methods=[], method_sweep=method)
    assert config.resolved_methods() == ["QIM_METHOD:step_scale=0.05", "QIM_METHOD:step_scale=0.2"]


def test_sweeps_are_validated_against_the_component():
    with pytest.raises(ValidationError, match="has no parameter"):
        _config(attack_sweep={"target": "awgn", "parameter": "bitrate_kbps", "values": [1, 2]})
    with pytest.raises(ValidationError, match="has no parameter"):
        _config(method_sweep={"target": "QIM_METHOD", "parameter": "alpha", "values": [1, 2]})
    with pytest.raises(ValidationError, match="distinct"):
        _config(attack_sweep={"target": "awgn", "parameter": "snr_db", "values": [10, 10]})


def test_threshold_crossing_interpolates_the_breakdown():
    assert threshold_crossing([(40, 0.0), (20, 0.05), (10, 0.15)], 0.1) == {"status": "crossed", "value": 15.0}
    assert threshold_crossing([(40, 0.0), (20, 0.02)], 0.1)["status"] == "never"
    assert threshold_crossing([(40, 0.3), (20, 0.4)], 0.1) == {"status": "always", "value": 40}


def test_robustness_curve_run():
    run = run_experiment(
        _config(
            methods=["QIM_METHOD", "LSB_METHOD"],
            attack_sweep={"target": "awgn", "parameter": "snr_db", "values": [40, 10, 0]},
        )
    )
    assert run.status == "completed", run.error
    curves = {curve["method"]: curve for curve in run.summary["curves"]}
    lsb = next(curve for name, curve in curves.items() if "LSB" in name)
    # LSB survives no additive noise at all: it breaks from the mildest setting.
    assert lsb["breakdown"]["status"] == "always"
    assert [point["value"] for point in lsb["points"]] == [40, 10, 0]
    assert run.summary["statistics"]["ber_sweep"]["available"]


def test_tradeoff_curve_run():
    run = run_experiment(
        _config(
            experiment_type=ExperimentType.TRADEOFF_CURVE,
            methods=["LSB_METHOD"],
            metrics=["SNR_METRIC"],
            method_sweep={"target": "QIM_METHOD", "parameter": "step_scale", "values": [0.02, 0.2]},
        )
    )
    assert run.status == "completed", run.error
    points = run.summary["points"]
    snr = [point["metrics"]["Signal-to-Noise Ratio (SNR)"]["estimate"] for point in points]
    assert snr[0] > snr[1]  # a larger quantisation step costs transparency
    assert [reference["spec"] for reference in run.summary["references"]] == ["LSB_METHOD"]
