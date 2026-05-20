from __future__ import annotations

from pathlib import Path

import pytest

from taf.audio.formats import AudioFileFormat, DecodeTarget
from taf.evaluation import (
    EvaluationConfig,
    EvaluationMessage,
    FailurePolicy,
    RandomMessageSpec,
    dump_config,
    load_config,
)
from taf.evaluation.yaml_loader import config_from_mapping
from taf.models.types import MethodType, MetricType


YAML_SAMPLE = """\
methods:
  - LSB_METHOD
  - DCT_B1_METHOD
metrics:
  - SNR_METRIC
  - PESQ_METRIC
target: FILES
formats:
  - wav
  - flac
output_dir: artifacts/run-2026-05-20
keep_files: false
overwrite: true
max_workers: 4
random_seed: 20260520
random_messages:
  - length: 16
    count: 2
    name_prefix: r16
random_message_lengths: [8, 32]
random_messages_per_length: 2
messages:
  - name: hello
    bits: [1, 0, 1, 1, 0, 1, 0, 0]
codec_options:
  flac:
    subtype: PCM_24
failure_policy: record
"""


def test_load_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(YAML_SAMPLE, encoding="utf-8")

    config = load_config(path)

    assert list(config.methods) == [MethodType.LSB_METHOD, MethodType.DCT_B1_METHOD]
    assert list(config.metrics) == [MetricType.SNR_METRIC, MetricType.PESQ_METRIC]
    assert config.target is DecodeTarget.FILES
    assert list(config.formats) == [AudioFileFormat.WAV, AudioFileFormat.FLAC]
    assert config.output_dir == Path("artifacts/run-2026-05-20")
    assert config.keep_files is False
    assert config.overwrite is True
    assert config.max_workers == 4
    assert config.random_seed == 20260520
    assert config.random_messages == (
        RandomMessageSpec(length=16, count=2, seed=None, name_prefix="r16"),
    )
    assert list(config.random_message_lengths) == [8, 32]
    assert config.random_messages_per_length == 2
    assert len(config.messages) == 1
    assert isinstance(config.messages[0], EvaluationMessage)
    assert config.messages[0].name == "hello"
    assert config.messages[0].bits == (1, 0, 1, 1, 0, 1, 0, 0)
    assert config.codec_options == {"flac": {"subtype": "PCM_24"}}
    assert config.failure_policy is FailurePolicy.RECORD


def test_partial_yaml_keeps_dataclass_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("max_workers: 8\n", encoding="utf-8")

    config = load_config(path)
    defaults = EvaluationConfig()

    assert config.max_workers == 8
    # Untouched fields should match dataclass defaults.
    assert config.target == defaults.target
    assert list(config.formats) == list(defaults.formats)
    assert config.failure_policy == defaults.failure_policy
    assert config.keep_files == defaults.keep_files


def test_empty_yaml_returns_default_config(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert load_config(path) == EvaluationConfig()


def test_yaml_roundtrip_preserves_round_tripable_fields(tmp_path: Path) -> None:
    original = EvaluationConfig(
        methods=(MethodType.LSB_METHOD,),
        metrics=(MetricType.SNR_METRIC,),
        target=DecodeTarget.FILES,
        formats=(AudioFileFormat.WAV, AudioFileFormat.FLAC),
        output_dir=tmp_path / "out",
        messages=(
            EvaluationMessage(name="m1", bits=(1, 0, 1)),
        ),
        random_messages=(RandomMessageSpec(length=4, count=1, seed=7),),
        random_seed=42,
        keep_files=False,
        overwrite=False,
        max_workers=3,
        codec_options={"flac": {"subtype": "PCM_16"}},
        failure_policy=FailurePolicy.RAISE,
    )

    target = tmp_path / "rt.yaml"
    dump_config(original, target)
    reloaded = load_config(target)

    assert reloaded == original


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Top-level YAML must be a mapping"):
        load_config(bad)


def test_classmethod_from_yaml_matches_module_loader(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(YAML_SAMPLE, encoding="utf-8")
    assert EvaluationConfig.from_yaml(path) == load_config(path)


def test_config_from_mapping_accepts_dict_directly() -> None:
    config = config_from_mapping({"target": "files", "formats": ["wav"], "max_workers": 2})
    assert config.target is DecodeTarget.FILES
    assert list(config.formats) == [AudioFileFormat.WAV]
    assert config.max_workers == 2
