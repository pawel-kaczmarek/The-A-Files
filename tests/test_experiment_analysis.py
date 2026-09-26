"""Statistical analysis, scenario summaries and provenance (taf.experiments)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")

from taf.evaluation.result import EvaluationRow, FailureKind
from taf.experiments import ExperimentConfig, ExperimentType, normalize_row, run_experiment
from taf.experiments.analysis import (
    FAILURE_IMPUTED_BER,
    cluster_bootstrap_ci,
    estimate,
    holm_adjust,
    paired_comparison,
    pareto_front,
)
from taf.experiments.csv_export import export_summary_csv
from taf.experiments.results import ExperimentResultRow, group_stats, metric_direction
from taf.experiments.scenarios import get_scenario


def _row(
    method: str,
    file_name: str,
    ber: float | None,
    *,
    attack: str | None = None,
    payload: int = 16,
    duration: float = 2.0,
    failure: str | None = None,
    metrics: dict[str, float] | None = None,
) -> ExperimentResultRow:
    return ExperimentResultRow(
        experiment_id="x",
        experiment_type="dataset_benchmark",
        timestamp=datetime.now(timezone.utc),
        file_name=file_name,
        file_path=file_name,
        duration_seconds=duration,
        method=method,
        payload_length=payload,
        attack=attack,
        bit_accuracy=None if ber is None else 1.0 - ber,
        ber=ber,
        decode_success=ber == 0.0,
        metrics=metrics or {},
        status="error" if failure else "ok",
        failure_kind=failure,
        error="failed" if failure else None,
    )


def _config(**overrides) -> ExperimentConfig:
    base = dict(
        experiment_type=ExperimentType.DATASET_BENCHMARK,
        name="t",
        dataset_id="vctk",
        methods=["LSB_METHOD", "QIM_METHOD"],
        payload_lengths=[16],
    )
    base.update(overrides)
    return ExperimentConfig(**base)


# ------------------------------------------------------------ estimation


def test_no_interval_from_a_single_cluster():
    assert cluster_bootstrap_ci([0.1, 0.2, 0.3], ["a", "a", "a"]) == (None, None)


def test_cluster_bootstrap_is_wider_than_treating_rows_as_independent():
    rng = np.random.default_rng(0)
    # Five files, 40 strongly correlated rows each.
    file_level = rng.normal(0.2, 0.1, size=5)
    values = [level + rng.normal(0, 0.005) for level in file_level for _ in range(40)]
    files = [f"f{index}" for index in range(5) for _ in range(40)]
    clustered = cluster_bootstrap_ci(values, files)
    independent = cluster_bootstrap_ci(values, list(range(len(values))))
    assert (clustered[1] - clustered[0]) > 3 * (independent[1] - independent[0])


def test_estimate_ignores_missing_values():
    result = estimate([0.0, None, 1.0, float("nan")], ["a", "b", "c", "d"])
    assert result["estimate"] == 0.5 and result["n"] == 2


# --------------------------------------------------- failures and BER


def test_failed_row_has_no_ber_and_keeps_its_reason():
    engine_row = EvaluationRow(
        input_path=Path("C:/data/a.wav"),
        method="LSB",
        message_name="random_000_len8_000",
        message_length=8,
        decode_mode="direct",
        format=None,
        success=False,
        error="message too long",
        failure_kind=FailureKind.OVER_CAPACITY,
        message_bits=[1] * 8,
        duration_seconds=2.0,
    )
    row = normalize_row(engine_row, experiment_id="x", experiment_type="t", dataset_id=None)
    assert row.ber is None and row.bit_accuracy is None
    assert row.failure_kind == FailureKind.OVER_CAPACITY
    assert row.payload_rate_bps == 4.0


def test_group_stats_separates_failures_from_bit_errors():
    rows = [_row("m", "a", 0.0), _row("m", "b", 0.2), _row("m", "c", None, failure="decode_error")]
    stats = group_stats(rows)
    assert stats["avg_ber"] == pytest.approx(0.1)  # completed trials only
    assert stats["completion_rate"] == pytest.approx(2 / 3)
    assert stats["failures"] == {"decode_error": 1}
    # Rankings see the failure at chance level, not as BER 1.0 and not as nothing.
    assert stats["avg_ber_imputed"] == pytest.approx((0.0 + 0.2 + FAILURE_IMPUTED_BER) / 3)


# ------------------------------------------------------------ comparison


def test_paired_comparison_finds_a_consistent_difference():
    rows = []
    for index in range(8):
        rows.append(_row("good", f"f{index}", 0.01 * index))
        rows.append(_row("bad", f"f{index}", 0.01 * index + 0.2))
        rows.append(_row("same", f"f{index}", 0.01 * index + 0.2))
    result = paired_comparison(rows, lambda r: r.ber, higher_is_better=False)
    assert result["available"]
    assert list(result["mean_ranks"])[0] == "good"
    assert result["omnibus"]["significant"]
    by_pair = {(p["a"], p["b"]): p for p in result["pairwise"]}
    assert by_pair[("bad", "good")]["better"] == "good"
    assert by_pair[("bad", "good")]["significant"]
    assert not by_pair[("bad", "same")]["significant"]
    assert result["critical_difference"] > 0


def test_paired_comparison_needs_shared_files():
    rows = [_row("a", "f1", 0.1), _row("b", "f2", 0.2)]
    assert paired_comparison(rows, lambda r: r.ber, higher_is_better=False)["available"] is False


def test_minimum_files_for_a_significant_pairwise_test():
    from scipy.stats import wilcoxon

    from taf.experiments.analysis import min_blocks_for_significance

    assert [min_blocks_for_significance(k) for k in (2, 3, 27)] == [6, 7, 14]
    # The bound is the exact test's smallest p-value times the number of pairs.
    for methods in (2, 3):
        needed = min_blocks_for_significance(methods)
        pairs = methods * (methods - 1) // 2
        best_p = wilcoxon(np.arange(1.0, needed + 1), np.zeros(needed)).pvalue
        worse_p = wilcoxon(np.arange(1.0, needed), np.zeros(needed - 1)).pvalue
        assert pairs * best_p < 0.05 <= pairs * worse_p


def test_holm_adjustment():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_pareto_front_keeps_trade_offs_and_drops_dominated_methods():
    scores = {
        "accurate": {"accuracy": 0.99, "quality": 10.0},
        "transparent": {"accuracy": 0.80, "quality": 40.0},
        "dominated": {"accuracy": 0.79, "quality": 9.0},
    }
    front = pareto_front(scores, {"accuracy": True, "quality": True})
    assert front["front"] == ["accurate", "transparent"]
    assert set(front["dominated_by"]["dominated"]) == {"accurate", "transparent"}


def test_metric_direction_comes_from_the_metric_not_its_name():
    assert metric_direction("Signal-to-Noise Ratio (SNR)") is True
    assert metric_direction("Weighted Spectral Slope (WSS)") is False
    assert metric_direction("BSS_EVAL version 4. [sdr]") is True
    assert metric_direction("BSS_EVAL version 4. [perm]") is None
    assert metric_direction("Speech-to-reverberation modulation energy ratio (SRMR) [cover]") is None


# -------------------------------------------------------------- scenarios


def test_benchmark_ranks_on_unattacked_rows():
    rows = []
    for index in range(6):
        file_name = f"f{index}"
        # "robust" is worse without attacks but survives them; the ranking
        # must not change with the attacks that were selected.
        rows += [_row("fragile", file_name, 0.0), _row("robust", file_name, 0.05)]
        rows += [
            _row("fragile", file_name, 0.5, attack="awgn"),
            _row("robust", file_name, 0.06, attack="awgn"),
        ]
    summary = get_scenario(ExperimentType.DATASET_BENCHMARK).summarize(
        rows, _config(attacks=["awgn"])
    )
    assert summary["method_ranking"][0] == "fragile"
    assert summary["statistics"]["ber_attacked"]["mean_ranks"]
    assert list(summary["statistics"]["ber_attacked"]["mean_ranks"])[0] == "robust"


def test_method_comparison_reports_a_front_and_no_default_weights():
    rows = []
    for index in range(6):
        rows += [
            _row("a", f"f{index}", 0.0, metrics={"Signal-to-Noise Ratio (SNR)": 20.0}),
            _row("b", f"f{index}", 0.1, metrics={"Signal-to-Noise Ratio (SNR)": 40.0}),
        ]
    config = _config(experiment_type=ExperimentType.METHOD_COMPARISON, methods=["LSB_METHOD", "QIM_METHOD"])
    summary = get_scenario(ExperimentType.METHOD_COMPARISON).summarize(rows, config)
    assert summary["pareto"]["front"] == ["a", "b"]
    assert summary["best_method"] is None  # a trade-off, not a winner
    assert summary["weights"] is None
    assert all("weighted_score" not in entry for entry in summary["comparison"])

    weighted = _config(
        experiment_type=ExperimentType.METHOD_COMPARISON,
        advanced_options={"weights": {"accuracy": 1.0}},
    )
    summary = get_scenario(ExperimentType.METHOD_COMPARISON).summarize(rows, weighted)
    assert summary["weighting"] and all("weighted_score" in e for e in summary["comparison"])


def test_capacity_is_measured_per_file_in_bits_per_second():
    rows = []
    for file_name, duration, limit in [("short", 1.0, 32), ("long", 4.0, 128)]:
        for payload in (16, 32, 64, 128, 256):
            if payload > limit:
                rows.append(_row("m", file_name, None, payload=payload, duration=duration, failure="over_capacity"))
            else:
                rows.append(_row("m", file_name, 0.0, payload=payload, duration=duration))
    config = _config(
        experiment_type=ExperimentType.EMBEDDING_CAPACITY,
        methods=["LSB_METHOD"],
        payload_lengths=[16, 32, 64, 128, 256],
    )
    summary = get_scenario(ExperimentType.EMBEDDING_CAPACITY).summarize(rows, config)
    (entry,) = summary["capacity_by_method"]
    assert entry["max_passing_payload"] == 32  # carried by every file
    assert entry["capacity_bits_median"] == pytest.approx((32 + 128) / 2)
    assert entry["capacity_bps_median"] == pytest.approx(32.0)  # 32 b/s on both files
    assert entry["over_capacity_rows"] == 4
    assert entry["censored_files"] == 0


def test_summary_csv_exports_nested_tables():
    rows = []
    for index in range(6):
        rows += [_row("a", f"f{index}", 0.0), _row("b", f"f{index}", 0.3)]
    summary = get_scenario(ExperimentType.DATASET_BENCHMARK).summarize(rows, _config())
    text = export_summary_csv(summary)
    assert "statistics.ber_baseline.pairwise" in text
    assert "[object" not in text and "{'" not in text


# ------------------------------------------------------ end-to-end runs


def test_run_resolves_the_seed_and_records_a_manifest():
    run = run_experiment(
        ExperimentConfig(
            experiment_type=ExperimentType.DATASET_BENCHMARK,
            name="manifest",
            dataset_id="vctk",
            file_limit=2,
            methods=["LSB_METHOD"],
            payload_lengths=[8],
        )
    )
    assert run.status == "completed"
    assert run.config.random_seed is not None
    manifest = run.manifest
    assert manifest["random_seed"] == run.config.random_seed
    assert len(manifest["inputs"]) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["inputs"])
    assert manifest["packages"]["numpy"]


def test_detectability_experiment_reports_uncertainty():
    run = run_experiment(
        ExperimentConfig(
            experiment_type=ExperimentType.DETECTABILITY,
            name="steg",
            dataset_id="vctk",
            file_limit=4,
            methods=["LSB_METHOD"],
            payload_lengths=[16],
            random_seed=3,
            advanced_options={"window_length": 8000},
        )
    )
    assert run.status == "completed", run.error
    (entry,) = run.summary["detectability"]
    assert entry["status"] == "ok"
    assert entry["accuracy_ci95_low"] <= entry["accuracy"] <= entry["accuracy_ci95_high"]
    assert 0.0 <= entry["p_value"] <= 1.0
    assert run.rows == []
