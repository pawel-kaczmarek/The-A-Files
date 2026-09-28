import json

import numpy as np
import pytest
import soundfile as sf

from taf.audio.metadata import describe_audio
from taf.corpora import scan_directory
from taf.experiments.runner import load_dataset_files, preview_experiment, run_experiment
from taf.experiments.schema import ExperimentConfig
from taf.experiments.inspector import resynthesize


def config(path, **kwargs):
    return ExperimentConfig(name="audio metadata", experiment_type="dataset_benchmark", dataset_path=str(path),
                            methods=["LSB_METHOD"], metrics=["MRSC_METRIC"], random_seed=1, max_workers=1, **kwargs)


@pytest.fixture
def audio(tmp_path):
    for index in range(4):
        path = tmp_path / str(index) / "same.wav"
        path.parent.mkdir()
        sf.write(path, np.random.default_rng(index).normal(0, .1, (16000, 2)), 16000, subtype="PCM_24")
    return tmp_path


def test_metadata_subset_and_mono_reproduction(audio):
    cfg = config(audio, selected_files=["1/same.wav", "2/same.wav"], file_limit=1, subset_seed=4, audio_category="music")
    assert preview_experiment(cfg).file_count == 1
    first, second = load_dataset_files(cfg), load_dataset_files(cfg)
    assert first[0].path == second[0].path
    assert first[0].samples.ndim == 1
    run = run_experiment(cfg)
    assert run.status == "completed", run.error
    row = run.rows[0]
    assert row.source_channels == 2 and row.channels == 1 and row.bit_depth == 24
    assert row.audio_category == "music" and row.preprocessing == ["arithmetic_mean_downmix"]
    assert resynthesize(run.config, row).reproduced


def test_ambiguous_names_and_channel_policy(audio):
    with pytest.raises(ValueError, match="ambiguous"):
        load_dataset_files(config(audio, selected_files=["same.wav"]))
    with pytest.raises(ValueError, match="channels"):
        load_dataset_files(config(audio, channel_policy="reject"))


def test_manifest_detects_content_drift(audio):
    manifest = scan_directory(audio)
    assert all(f["bit_depth"] == 24 and f["sha256"] for f in manifest["files"])
    (audio / "manifest.json").write_text(json.dumps(manifest))
    load_dataset_files(config(audio))
    sf.write(audio / "0/same.wav", np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_dataset_files(config(audio))


def test_float_bit_depth_unknown(tmp_path):
    path = tmp_path / "float.wav"
    sf.write(path, np.zeros(160), 16000, subtype="FLOAT")
    assert describe_audio(path)["bit_depth"] is None


def test_prepared_subsets_record_native_audio_and_reproduce(audio, tmp_path):
    from dataclasses import replace
    from taf.corpora import SubsetRule, get_corpus, prepare_subset

    corpus = replace(get_corpus("vctk"), audio_glob="**/*.wav", path_filter=None)
    rule = SubsetRule(max_files=2, target_sample_rate=8000, seed=123)
    first = prepare_subset(corpus, audio, tmp_path / "first", rule)
    second = prepare_subset(corpus, audio, tmp_path / "second", rule)
    assert [f["sha256"] for f in first["files"]] == [f["sha256"] for f in second["files"]]
    for file in first["files"]:
        assert file["source_audio"]["channels"] == 2
        assert file["source_audio"]["bit_depth"] == 24
        assert file["source_sha256"]
        assert file["channels"] == 1 and file["bit_depth"] == 16 and file["sample_rate"] == 8000


def test_exported_run_pins_subset_and_digests(audio):
    run = run_experiment(config(audio, file_limit=3, subset_seed=12))
    selected = run.config.selected_files
    assert len(selected) == 3 and selected[0] in run.config.selected_file_sha256
    sf.write(audio / "unrelated.wav", np.ones(16000) * .1, 16000)
    replay = run_experiment(run.config)
    assert replay.status == "completed" and replay.config.selected_files == selected
    sf.write(audio / selected[0], np.zeros(16000), 16000)
    changed = run_experiment(run.config)
    assert changed.status == "failed" and "SHA-256 mismatch" in changed.error


def test_library_scan_preserves_annotations_and_checks_hashes(audio):
    manifest = scan_directory(audio)
    manifest["category"] = "music"
    manifest["files"][0].update(source="Original archive", speaker="s01", category="speech")
    (audio / "manifest.json").write_text(json.dumps(manifest))
    scanned = scan_directory(audio)
    assert scanned["category"] == "music"
    assert scanned["files"][0]["source"] == "Original archive"
    files = load_dataset_files(config(audio))
    assert files[0].metadata["category"] == "speech"
    assert files[1].metadata["category"] == "music"
    sf.write(audio / "0/same.wav", np.zeros(16000), 16000)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        scan_directory(audio)
