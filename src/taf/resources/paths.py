from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

RESOURCE_PACKAGE = "taf.resources"
PACKAGED_DATASET_GROUPS: dict[str, tuple[str, ...]] = {
    "vctk": ("datasets", "VCTK", "16"),
    "librispeech": ("datasets", "LibriSpeech", "142345"),
}


def resource_files(*parts: str):
    resource = files(RESOURCE_PACKAGE)
    for part in parts:
        resource = resource.joinpath(part)

    if not resource.is_file() and not resource.is_dir():
        joined = "/".join(parts)
        raise FileNotFoundError(f"Packaged resource not found: {joined}")

    return resource


@contextmanager
def resource_path(*parts: str) -> Iterator[Path]:
    with as_file(resource_files(*parts)) as path:
        yield path


def read_text(*parts: str, encoding: str = "utf-8") -> str:
    return resource_files(*parts).read_text(encoding=encoding)


def read_bytes(*parts: str) -> bytes:
    return resource_files(*parts).read_bytes()


def read_json(*parts: str) -> dict[str, Any]:
    return json.loads(read_text(*parts))


@contextmanager
def example_wav_path() -> Iterator[Path]:
    with resource_path("audio", "example.wav") as path:
        yield path


@contextmanager
def mosnet_model_path() -> Iterator[Path]:
    with resource_path("models", "mosnet", "cnn_blstm.h5") as path:
        yield path


def packaged_dataset_manifest() -> dict[str, tuple[str, ...]]:
    return PACKAGED_DATASET_GROUPS.copy()


@contextmanager
def packaged_dataset_audio_paths() -> Iterator[dict[str, list[Path]]]:
    with ExitStack() as stack:
        paths: dict[str, list[Path]] = {}
        for group_name, parts in PACKAGED_DATASET_GROUPS.items():
            group_resource = resource_files(*parts)
            flac_resources = sorted(
                child for child in group_resource.iterdir()
                if child.is_file() and child.name.endswith(".flac")
            )
            paths[group_name] = [
                stack.enter_context(as_file(resource))
                for resource in flac_resources
            ]
        yield paths
