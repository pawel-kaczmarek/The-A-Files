import math

import pytest
from pydantic import ValidationError

from taf.experiments.payloads import PayloadSpec, bits_for_rate, payload_digest
from taf.experiments.runner import preview_experiment, run_experiment
from taf.experiments.schema import ExperimentConfig
from taf.experiments.csv_export import rows_to_dataframe


def config(**kwargs):
    return ExperimentConfig(name="payload test", experiment_type="dataset_benchmark", dataset_id="vctk",
                            file_limit=2, methods=["LSB_METHOD"], random_seed=42, max_workers=1, **kwargs)


def test_utf8_binary_and_unaligned_bits():
    text = PayloadSpec(kind="text", value="ą\x00")
    binary = PayloadSpec(kind="binary", value="c48500")
    assert text.bits() == binary.bits()
    assert len(text.bits()) == 24
    assert PayloadSpec(kind="bits", value="001").bits() == (0, 0, 1)
    assert payload_digest([0, 1]) != payload_digest([0, 1, 0])


@pytest.mark.parametrize("kind,value", [("bits", "102"), ("bits", "0 1"), ("text", ""), ("binary", "0"), ("binary", "gg"), ("random", "hello")])
def test_reject_malformed_payload(kind, value):
    with pytest.raises(ValidationError):
        PayloadSpec(kind=kind, value=value)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_rates(rate):
    with pytest.raises(ValidationError):
        config(payload_rates_bps=[rate])


def test_rate_bounds_and_explicit_content():
    assert bits_for_rate(4, 10001, 16000) == 2
    with pytest.raises(ValueError):
        bits_for_rate(.1, 100, 16000)
    with pytest.raises(ValueError):
        bits_for_rate(9000, 16000, 16000)
    with pytest.raises(ValidationError):
        config(payload=PayloadSpec(kind="text", value="abc"), payload_rates_bps=[10])


def test_rate_sweep_reproducible_and_exported():
    cfg = config(payload_rates_bps=[8, 16], repetitions=2)
    run, again = run_experiment(cfg), run_experiment(cfg)
    assert run.status == "completed", run.error
    assert len(run.rows) == preview_experiment(cfg).estimated_result_rows == 8
    assert [r.payload_sha256 for r in run.rows] == [r.payload_sha256 for r in again.rows]
    for row in run.rows:
        assert row.decode_success
        assert row.payload_length == math.floor(row.requested_payload_rate_bps * row.duration_seconds)
        assert row.payload_rate_bps <= row.requested_payload_rate_bps
        assert row.exact_goodput_bps == row.payload_rate_bps
        assert row.payload_bits_per_sample == row.payload_length / row.sample_count
        assert row.encode_rtf == row.encode_time_seconds / row.duration_seconds
        assert row.bit_depth == 16 and row.audio_sha256 and row.payload_seed is not None
    columns = rows_to_dataframe(run.rows).columns
    for name in ("payload_rate_bps", "payload_sha256", "encode_rtf", "bit_depth", "method_parameters", "failure_kind"):
        assert name in columns


def test_explicit_payload_uses_exact_size():
    run = run_experiment(config(payload=PayloadSpec(kind="binary", value="0001ff"), payload_lengths=[16, 32], repetitions=2))
    assert len(run.rows) == 4
    assert all(r.payload_length == 24 and r.payload_kind == "binary" and r.decode_success for r in run.rows)
    assert len({r.payload_sha256 for r in run.rows}) == 1


def test_unused_length_list_may_be_empty():
    assert config(payload_lengths=[], payload_rates_bps=[8]).payload_variant_count() == 1
    assert config(payload_lengths=[], payload={"kind": "bits", "value": "001"}).payload_variant_count() == 1
    with pytest.raises(ValidationError):
        config(payload_lengths=[])
