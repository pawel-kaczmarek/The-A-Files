"""Stable relative identities, recorded preprocessing and manifest validation."""

import json
from pathlib import Path

import numpy as np

from taf.corpora.prepare import sha256_of


def file_identity(file, config, directory=None):
    path = Path(file.path if hasattr(file, "path") else file)
    if directory:
        return path.relative_to(directory).as_posix()
    return "/".join(path.parts[-3:]) if config.dataset_id == "all" else path.name


def select_files(files, config, directory=None):
    def identity(file):
        return file_identity(file, config, directory)

    ordered = sorted(files, key=identity)
    if config.selected_files:
        selected = []
        for name in dict.fromkeys(config.selected_files):
            matches = [file for file in ordered if identity(file) == name]
            if not matches:
                matches = [file for file in ordered if Path(identity(file)).name == name]
            if len(matches) != 1:
                raise ValueError(f"Selected file {name!r} is missing or ambiguous; use its relative path.")
            selected.extend(matches)
        ordered = list({identity(file): file for file in selected}.values())
        if not config.selected_file_sha256:
            ordered.sort(key=identity)
    # A recorded run already contains the sampled order. Applying the subset
    # permutation again would change file-based train/test splits on replay.
    if config.subset_seed is not None and not config.selected_file_sha256:
        order = np.random.default_rng(config.subset_seed).permutation(len(ordered))
        ordered = [ordered[index] for index in order]
    return ordered[:config.file_limit] if config.file_limit is not None else ordered


def prepare_inputs(files, config, directory=None, manifest=None):
    if manifest is None and directory and (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest = manifest or {}
    entries = {entry["file"]: entry for entry in manifest.get("files", [])}
    for file in files:
        path = Path(file.path)
        identity = file_identity(file, config, directory)
        entry = entries.get(identity, {})
        digest = sha256_of(path)
        expected = config.selected_file_sha256.get(identity)
        if expected and expected != digest:
            raise ValueError(f"Recorded run SHA-256 mismatch for {identity}.")
        if entry.get("sha256") and entry["sha256"] != digest:
            raise ValueError(f"Dataset manifest SHA-256 mismatch for {identity}.")
        category = config.audio_category or entry.get("category") or manifest.get("category")
        if category is None and config.dataset_id in ("vctk", "librispeech", "all"):
            category = "speech"
        file.metadata.update({
            "file_id": identity, "sha256": digest, "category": category or "unknown",
            "source": config.audio_source or entry.get("source") or manifest.get("source") or config.dataset_id,
            "speaker": entry.get("speaker"), "preprocessing": [],
        })
        if file.samples.ndim > 1:
            if config.channel_policy == "reject":
                raise ValueError(f"{identity} has {file.samples.shape[1]} channels; mono input required.")
            file.samples = file.samples.mean(axis=1)
            file.metadata["preprocessing"].append("arithmetic_mean_downmix")
    return files
