"""The shared evaluation block (taf.experiments.scenarios.evaluation) and its estimators."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

from taf.experiments import ExperimentConfig, ExperimentType
from taf.experiments.analysis import box_summary, paired_difference, stratified_spearman
from taf.experiments.reporting import build_report
from taf.experiments.results import ExperimentResultRow, distribution_stats
from taf.experiments.scenarios import summarize_for_scenario
from taf.experiments.scenarios.evaluation import EVALUATION_VERSION, attack_level

FILES = [f"{index}.flac" for index in range(8)]
SNR = "Signal-to-Noise Ratio (SNR)"


def _row(
    method: str,
    file_name: str,
    ber: float | None,
    *,
    attack: str | None = None,
    payload: int = 32,
    failure: str | None = None,
    metrics: dict[str, float] | None = None,
    encode: float = 0.02,
    decode: float = 0.01,
) -> ExperimentResultRow:
    return ExperimentResultRow(
        experiment_id="x",
        experiment_type="research_experiment",
        timestamp=datetime.now(timezone.utc),
        file_name=file_name,
        file_path=file_name,
        duration_seconds=2.0,
        method=method,
        payload_length=payload,
        payload_rate_bps=payload / 2.0,
        attack=attack,
        bit_accuracy=None if ber is None else 1.0 - ber,
        ber=ber,
        decode_success=ber == 0.0,
        metrics=metrics or {},
        encode_time_seconds=encode,
        decode_time_seconds=decode,
        status="error" if failure else "ok",
        failure_kind=failure,
        error="failed" if failure else None,
    )


def _config(**overrides) -> ExperimentConfig:
    base = dict(
        experiment_type=ExperimentType.RESEARCH_EXPERIMENT,
        name="t",
        dataset_id="vctk",
        methods=["LSB_METHOD", "QIM_METHOD"],
        payload_lengths=[32],
        attacks=["awgn@mild", "awgn@strong"],
        max_workers=1,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def _robustness_rows() -> list[ExperimentResultRow]:
    """ROBUST keeps BER 0; FRAGILE degrades with the strength of awgn."""
    rows = []
    for index, name in enumerate(FILES):
        wobble = index * 0.004
        for method, mild, strong in (("ROBUST", 0.0, 0.0), ("FRAGILE", 0.10 + wobble, 0.30 + wobble)):
            rows.append(_row(method, name, 0.0, metrics={SNR: 40.0 - index}))
            rows.append(_row(method, name, mild, attack="awgn@mild"))
            rows.append(_row(method, name, strong, attack="awgn@strong"))
    return rows


# ---------------------------------------------------------------- estimators


def test_distribution_stats_reports_quartiles():
    stats = distribution_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (stats["q1"], stats["median"], stats["q3"], stats["iqr"]) == (2.0, 3.0, 4.0, 2.0)
    assert stats["min"] == 1.0 and stats["max"] == 5.0


def test_box_summary_whiskers_stop_at_the_fences():
    box = box_summary([0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.5])
    assert box["whisker_high"] < 0.5
    assert box["outliers"] == 1
    assert box["n"] == 7


def test_paired_difference_uses_only_shared_files():
    delta = paired_difference({"a": 0.3, "b": 0.5, "c": 0.9}, {"a": 0.1, "b": 0.1})
    assert delta["estimate"] == pytest.approx(0.3)
    assert delta["clusters"] == 2


def test_stratified_spearman_detects_a_within_file_trend():
    rng = np.random.default_rng(1)
    x, y, strata = [], [], []
    for file_index in range(8):
        offset = rng.normal(0, 1)  # files differ; the trend is inside each file
        for level in range(4):
            x.append(level)
            y.append(offset + 0.5 * level + rng.normal(0, 0.05))
            strata.append(file_index)
    result = stratified_spearman(x, y, strata)
    assert result["available"]
    assert result["p_value"] < 0.01
    assert result["n_files"] == 8 and result["levels"] == 4
    assert result == stratified_spearman(x, y, strata)  # seeded


def test_stratified_spearman_finds_no_trend_in_noise():
    rng = np.random.default_rng(2)
    strata = [index // 4 for index in range(40)]
    result = stratified_spearman([index % 4 for index in range(40)], rng.normal(size=40).tolist(), strata)
    assert result["p_value"] > 0.05


def test_stratified_spearman_needs_two_levels():
    result = stratified_spearman([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4], ["a", "a", "b", "b"])
    assert not result["available"]


def test_attack_levels_are_ordered_by_severity():
    assert attack_level(None)["rank"] == 0
    assert attack_level("mp3@mild")["rank"] < attack_level("mp3@strong")["rank"]
    assert attack_level("mp3@strong")["family"] == "mp3"
    assert attack_level("mp3:bitrate=64")["rank"] is None
    assert attack_level("awgn:snr_db=10", {"awgn:snr_db=10": (2, 10)}) == {"family": "awgn", "level": "10", "rank": 3}


# ----------------------------------------------------------- evaluation block


def test_evaluation_block_is_attached_and_versioned():
    summary = summarize_for_scenario(_robustness_rows(), _config())
    evaluation = summary["evaluation"]
    assert evaluation["version"] == EVALUATION_VERSION
    settings = evaluation["settings"]
    for key in ("confidence", "bootstrap_resamples", "bootstrap_seed", "correction", "permutations", "permutation_seed"):
        assert key in settings
    # Comparisons join the existing statistics tab instead of a second subsystem.
    assert set(summary["statistics"]["per_attack"]) == {"awgn@mild", "awgn@strong"}
    assert "metrics" in summary["statistics"]


def test_recovery_rates_and_ber_resolution():
    evaluation = summarize_for_scenario(_robustness_rows(), _config())["evaluation"]
    fragile = next(entry for entry in evaluation["methods"] if entry["method"] == "FRAGILE")
    assert fragile["clean"]["recovery"]["exact"]["estimate"] == 1.0
    assert fragile["attacked"]["recovery"]["le_5pct"]["estimate"] == 0.0
    assert fragile["attacked"]["ber"]["q3"] >= fragile["attacked"]["ber"]["median"]
    # With 32-bit payloads BER moves in steps of 1/32, so <= 1% means exactly 0.
    assert evaluation["resolution"]["le_1pct_equals_exact"] is True


def test_attack_increase_is_paired_by_file_and_ranked():
    evaluation = summarize_for_scenario(_robustness_rows(), _config())["evaluation"]
    destructive = evaluation["most_destructive"]["FRAGILE"]
    assert destructive[0]["attack"] == "awgn@strong"
    assert destructive[0]["delta_ber"]["estimate"] == pytest.approx(0.30 + 0.004 * 3.5)
    strong = next(attack for attack in evaluation["attacks"] if attack["attack"] == "awgn@strong")
    assert strong["most_resistant"][0] == "ROBUST"


def test_severity_trend_and_facts_come_from_the_data():
    evaluation = summarize_for_scenario(_robustness_rows(), _config())["evaluation"]
    family = evaluation["severity"][0]
    assert family["family"] == "awgn" and family["levels"] == ["clean", "mild", "strong"]
    fragile = next(curve for curve in family["curves"] if curve["method"] == "FRAGILE")
    assert fragile["trend"]["rho"] > 0.8 and fragile["trend"]["p_value"] < 0.01
    robust = next(curve for curve in family["curves"] if curve["method"] == "ROBUST")
    assert not robust["trend"]["available"]  # BER never varies

    kinds = {fact["kind"] for fact in evaluation["facts"]}
    assert {"largest_ber_increase", "stable_under_attack", "correlation"} <= kinds
    stable = [fact for fact in evaluation["facts"] if fact["kind"] == "stable_under_attack"]
    assert all(fact["methods"] == ["ROBUST"] for fact in stable)
    increase = next(fact for fact in evaluation["facts"] if fact["kind"] == "largest_ber_increase")
    assert increase["method"] == "FRAGILE" and increase["ci95_low"] > 0


def test_failures_are_not_recovered_and_are_reported():
    rows = _robustness_rows() + [_row("FRAGILE", FILES[0], None, attack="awgn@strong", failure="decode_error")]
    evaluation = summarize_for_scenario(rows, _config())["evaluation"]
    fact = next(fact for fact in evaluation["facts"] if fact["kind"] == "failures")
    assert fact["method"] == "FRAGILE" and fact["by_kind"] == {"decode_error": 1}


def test_payload_and_runtime_sections():
    rows = []
    for index, name in enumerate(FILES):
        for payload in (16, 64, 256):
            ber = 0.0 if payload < 256 else 0.05 + index * 0.001
            rows.append(_row("M", name, ber, payload=payload, metrics={SNR: 50.0 - payload / 10}, encode=payload * 1e-4))
    evaluation = summarize_for_scenario(rows, _config(methods=["LSB_METHOD"], attacks=[], payload_lengths=[16, 64, 256]))[
        "evaluation"
    ]
    assert evaluation["payload"]["available"] and evaluation["payload"]["levels"] == [16, 64, 256]
    point = evaluation["payload"]["per_method"][0]["points"][0]
    assert point["payload_bps"]["mean"] == pytest.approx(8.0)
    runtime = evaluation["runtime"]["per_method"][0]
    assert runtime["real_time_factor"]["mean"] == pytest.approx(runtime["total_seconds"]["mean"] / 2.0)
    tests = {(test["x"], test["y"]): test for test in evaluation["correlations"]}
    assert tests[("payload_bits", f"metric: {SNR}")]["rho"] == pytest.approx(-1.0)
    assert tests[("payload_bits", "total_seconds")]["available"]


def test_runtime_is_not_compared_with_parallel_workers():
    summary = summarize_for_scenario(_robustness_rows(), _config(max_workers=2))
    assert summary["evaluation"]["runtime"]["comparable"] is False
    assert "runtime_total" not in summary["statistics"]


def test_report_lists_the_findings():
    summary = summarize_for_scenario(_robustness_rows(), _config())
    text = build_report("t", _config(), {}, summary, "markdown")
    assert "## Findings" in text
    latex = build_report("t", _config(), {}, summary, "latex")
    assert "\\subsection*{Findings}" in latex and "ρ" not in latex


def test_trend_on_the_curve_carries_the_holm_adjusted_p_value():
    evaluation = summarize_for_scenario(_robustness_rows(), _config())["evaluation"]
    fragile = next(curve for curve in evaluation["severity"][0]["curves"] if curve["method"] == "FRAGILE")
    listed = next(test for test in evaluation["correlations"] if test["method"] == "FRAGILE" and test["x"] == "attack_strength")
    assert fragile["trend"]["p_holm"] == listed["p_holm"]
