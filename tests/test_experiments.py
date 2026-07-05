"""Tests for the pure-Python experiment engine (taf.experiments)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")

from pydantic import ValidationError

from taf.evaluation.result import EvaluationRow
from taf.experiments import (
    ExperimentConfig,
    ExperimentType,
    bit_accuracy,
    bit_error_rate,
    export_detailed_csv,
    export_summary_csv,
    make_export_filename,
    normalize_row,
    preview_experiment,
    run_experiment,
)
from taf.experiments.runner import validate_config
from taf.experiments.scenarios import get_scenario, validate_for_scenario


def _config(**overrides) -> ExperimentConfig:
    base = dict(
        experiment_type=ExperimentType.DATASET_BENCHMARK,
        name="test",
        dataset_id="example",
        methods=["LSB_METHOD"],
        payload_lengths=[8],
    )
    base.update(overrides)
    return ExperimentConfig(**base)


# ---------------------------------------------------------------- schema


def test_config_requires_method():
    with pytest.raises(ValidationError, match="At least one method"):
        _config(methods=[])


def test_config_requires_dataset_source():
    with pytest.raises(ValidationError, match="dataset_id or dataset_path"):
        _config(dataset_id=None)


def test_config_rejects_unknown_selections():
    with pytest.raises(ValidationError, match="Unknown method"):
        _config(methods=["NOPE"])
    with pytest.raises(ValidationError, match="Unknown metric"):
        _config(metrics=["NOPE"])
    with pytest.raises(ValidationError, match="Unknown attack"):
        _config(attacks=["NOPE"])
    with pytest.raises(ValidationError, match="Unknown dataset"):
        _config(dataset_id="nope")


def test_payload_length_validation():
    with pytest.raises(ValidationError, match="between 4 and 120"):
        _config(payload_lengths=[2])
    with pytest.raises(ValidationError, match="between 4 and 120"):
        _config(payload_lengths=[500])
    with pytest.raises(ValidationError, match="unique"):
        _config(payload_lengths=[8, 8])
    with pytest.raises(ValidationError, match="At least one payload length"):
        _config(payload_lengths=[])


def test_scenario_specific_validation():
    robustness = _config(experiment_type=ExperimentType.ATTACK_ROBUSTNESS)
    assert any("attack" in p.lower() for p in validate_for_scenario(robustness))

    quality = _config(experiment_type=ExperimentType.PERCEPTUAL_QUALITY)
    assert any("metric" in p.lower() for p in validate_for_scenario(quality))

    comparison = _config(experiment_type=ExperimentType.METHOD_COMPARISON)
    assert any("two methods" in p for p in validate_for_scenario(comparison))

    capacity = _config(experiment_type=ExperimentType.EMBEDDING_CAPACITY)
    assert any("two payload lengths" in p for p in validate_for_scenario(capacity))

    assert get_scenario("dataset_benchmark").title == "Dataset Benchmark"


def test_validate_config_flags_missing_local_dataset():
    config = _config(dataset_id=None, dataset_path="Z:/does/not/exist")
    assert any("Dataset not found" in p for p in validate_config(config))


# ------------------------------------------------------------ bit metrics


def test_bit_accuracy_and_ber():
    assert bit_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert bit_accuracy([1, 0, 1, 0], [0, 1, 0, 1]) == 0.0
    assert bit_accuracy([1, 1, 1, 1], [1, 1, 0, 0]) == 0.5
    # Missing decoded bits count as errors.
    assert bit_accuracy([1, 1, 1, 1], [1, 1]) == 0.5
    assert bit_accuracy([1, 1], None) == 0.0
    assert bit_error_rate([1, 1, 1, 1], [1, 1, 0, 0]) == 0.5


# ------------------------------------------------------- normalization


def _engine_row(**overrides) -> EvaluationRow:
    base = dict(
        input_path=Path("C:/data/sample.wav"),
        method="Standard LSB coding method",
        message_name="random_000_len8_002",
        message_length=8,
        decode_mode="direct",
        format=None,
        success=False,
        message_bits=[1, 0, 1, 0, 1, 0, 1, 0],
        decoded_message=[1, 0, 1, 0, 0, 1, 1, 0],
        sample_rate=16000,
        duration_seconds=0.35,
        encode_time_seconds=0.01,
        decode_time_seconds=0.02,
        attack_time_seconds=0.005,
        metrics={"SNR": 21.5, "BROKEN": float("nan")},
    )
    base.update(overrides)
    return EvaluationRow(**base)


def test_normalize_row():
    row = normalize_row(
        _engine_row(),
        experiment_id="exp1",
        experiment_type="dataset_benchmark",
        dataset_id="example",
    )
    assert row.file_name == "sample.wav"
    assert row.payload_length == 8
    assert row.repetition == 2
    assert row.message_bits == "10101010"
    assert row.decoded_bits == "10100110"
    assert row.bit_accuracy == 0.75
    assert row.ber == 0.25
    assert row.decode_success is False
    assert row.metrics["SNR"] == 21.5
    assert row.metrics["BROKEN"] is None  # NaN normalized away
    assert row.total_time_seconds == pytest.approx(0.035)
    assert row.status == "ok"


def test_normalize_row_error_status():
    row = normalize_row(
        _engine_row(error="encode blew up", decoded_message=None),
        experiment_id="exp1",
        experiment_type="dataset_benchmark",
        dataset_id="example",
    )
    assert row.status == "error"
    assert row.error == "encode blew up"
    assert row.decoded_bits is None


# ------------------------------------------------------------- preview


def test_preview_counts_and_warnings():
    config = _config(
        methods=["LSB_METHOD", "NORM_SPACE_METHOD"],
        payload_lengths=[8, 16],
        repetitions=3,
        attacks=["additive_noise", "resample"],
        metrics=["SNR_METRIC"],
    )
    plan = preview_experiment(config)
    assert plan.file_count == 1  # example dataset
    assert plan.attack_variant_count == 3  # baseline + 2 attacks
    assert plan.encode_operations == 1 * 2 * 2 * 3
    assert plan.estimated_result_rows == plan.encode_operations * 3
    assert plan.estimated_metric_calculations == plan.estimated_result_rows
    assert any(w.code == "attack_changes_signal" for w in plan.warnings)


def test_preview_flags_validation_problems():
    config = _config(experiment_type=ExperimentType.ATTACK_ROBUSTNESS, attacks=[])
    plan = preview_experiment(config)
    assert any(w.code == "validation" for w in plan.warnings)


# ------------------------------------------------------------ csv export


def test_csv_export_columns_and_filename():
    rows = [
        normalize_row(
            _engine_row(),
            experiment_id="exp1",
            experiment_type="dataset_benchmark",
            dataset_id="example",
        )
    ]
    csv_text = export_detailed_csv(rows)
    header = csv_text.splitlines()[0].split(",")
    for column in ("experiment_id", "method", "payload_length", "ber", "bit_accuracy", "status"):
        assert column in header
    assert "metric:SNR" in header
    assert len(csv_text.splitlines()) == 2

    name = make_export_filename("dataset_benchmark", "ab12", "detailed")
    assert name.startswith("dataset_benchmark_ab12_detailed_")
    assert name.endswith(".csv")

    assert export_detailed_csv([]).strip() != ""  # header-only export still valid


# ----------------------------------------------------- end-to-end engine


def test_run_experiment_records_failures_per_row():
    # DSSS needs long inputs; on the tiny example file it must fail per-row
    # while LSB succeeds — the run itself completes.
    config = _config(
        methods=["LSB_METHOD", "DSSS_METHOD"],
        payload_lengths=[8],
        random_seed=42,
    )
    run = run_experiment(config)
    assert run.status == "completed"
    lsb = next(row for row in run.rows if "LSB" in (row.method_type or row.method))
    dsss = next(row for row in run.rows if "DSSS" in (row.method_type or row.method))
    assert lsb.status == "ok" and lsb.decode_success is True
    assert dsss.status == "error" and dsss.error
    assert run.summary["overall"]["rows"] == 2


def test_run_experiment_attack_robustness_summary():
    config = _config(
        experiment_type=ExperimentType.ATTACK_ROBUSTNESS,
        methods=["LSB_METHOD", "NORM_SPACE_METHOD"],
        attacks=["amplitude_scaling"],
        payload_lengths=[8],
        random_seed=42,
    )
    run = run_experiment(config)
    assert run.status == "completed"
    assert len(run.rows) == 4  # 2 methods x (baseline + 1 attack)
    assert run.summary["most_robust_method"]
    assert run.summary["worst_attack"] == "amplitude_scaling"
    matrix = run.summary["matrix"]
    assert {(cell["method"], cell["attack"]) for cell in matrix} == {
        ("Standard LSB coding method", None),
        ("Standard LSB coding method", "amplitude_scaling"),
        ("Norm space method", None),
        ("Norm space method", "amplitude_scaling"),
    }
    assert export_summary_csv(run.summary).strip()


def test_run_experiment_fails_cleanly_on_bad_dataset():
    config = _config(dataset_id=None, dataset_path="Z:/missing")
    run = run_experiment(config)
    assert run.status == "failed"
    assert "Dataset not found" in (run.error or "")
    assert run.rows == []
