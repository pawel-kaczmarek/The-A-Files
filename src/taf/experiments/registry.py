"""Discovery of methods, metrics, attacks and datasets for experiments.

Thin wrappers over the existing factories/helpers so that scripts, the API
and the UI all see the same inventory.
"""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taf.experiments.schema import UPLOAD_DATASET_PREFIX

_DEFAULT_SAMPLE_RATE = 16000

# Methods/metrics that require the optional TensorFlow extra ("ai").
TENSORFLOW_METHODS = {"FGAS_METHOD"}
TENSORFLOW_METRICS = {"AI_MOSNET_METRIC"}

# Methods that reserve floor(len/8192) frames and need >= 8 frames, i.e. need
# long inputs (see tests/conftest.py). Used only to produce preview warnings.
LONG_INPUT_METHODS = {"ECHO_METHOD", "DSSS_METHOD"}

_ATTACK_DESCRIPTIONS = {
    "additive_noise": "Adds Gaussian noise to the signal.",
    "amplitude_scaling": "Multiplies all samples by a constant factor.",
    "cut_random_samples": "Zeroes a number of randomly chosen samples.",
    "flip_random_samples": "Inverts the sign of randomly chosen samples.",
    "frequency_filter": "Removes a single frequency bin via FFT filtering.",
    "low_pass_filter": "Butterworth low-pass filter.",
    "pitch_shift": "Shifts pitch by n semitone steps (librosa).",
    "resample": "Resamples the signal to a different sample rate.",
    "time_stretch": "Stretches/compresses the signal in time (librosa).",
}


@dataclass(frozen=True)
class AttackParameter:
    name: str
    default: Any


@dataclass(frozen=True)
class AttackSpec:
    name: str
    class_name: str
    description: str
    parameters: list[AttackParameter] = field(default_factory=list)
    changes_length_or_rate: bool = False


def list_methods() -> list[dict[str, Any]]:
    from taf.ui.helpers import discover_methods

    return [
        {
            "name": row["name"],
            "class_name": row["class"],
            "description": row["type"],
            "requires_tensorflow": row["name"] in TENSORFLOW_METHODS,
            "needs_long_input": row["name"] in LONG_INPUT_METHODS,
        }
        for row in discover_methods(_DEFAULT_SAMPLE_RATE)
    ]


def list_metrics() -> list[dict[str, Any]]:
    from taf.ui.helpers import discover_metrics

    return [
        {
            "name": row["name"],
            "class_name": row["class"],
            "category": row["category"],
            "requires_tensorflow": row["name"] in TENSORFLOW_METRICS,
            # All packaged metrics compare original vs processed samples and
            # require both signals to share length/sample rate.
            "compares_original": True,
            "supports_attacked_audio": True,
        }
        for row in discover_metrics()
    ]


def list_attacks() -> list[AttackSpec]:
    from taf.attacks.attacks import CorruptedWavFile

    specs: list[AttackSpec] = []
    for name, member in sorted(inspect.getmembers(CorruptedWavFile, predicate=inspect.isfunction)):
        if name.startswith("_") or name not in CorruptedWavFile.__dict__:
            continue
        parameters = [
            AttackParameter(name=param.name, default=param.default)
            for param in inspect.signature(member).parameters.values()
            if param.name != "self"
        ]
        specs.append(
            AttackSpec(
                name=name,
                class_name=CorruptedWavFile.__name__,
                description=_ATTACK_DESCRIPTIONS.get(name, ""),
                parameters=parameters,
                changes_length_or_rate=name in {"resample", "time_stretch", "pitch_shift"},
            )
        )
    return specs


def tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def method_descriptions() -> dict[str, str]:
    """Map of human method labels (``method.type()``) back to enum names."""
    return {row["description"]: row["name"] for row in list_methods() if row["description"]}


def list_datasets() -> list[dict[str, Any]]:
    from taf.resources.paths import packaged_dataset_audio_paths

    datasets: list[dict[str, Any]] = [
        {"id": "example", "label": "example", "kind": "packaged", "file_count": 1, "formats": ["wav"]}
    ]
    with packaged_dataset_audio_paths() as groups:
        for group_name, paths in groups.items():
            datasets.append(
                {
                    "id": group_name,
                    "label": group_name,
                    "kind": "packaged",
                    "file_count": len(paths),
                    "formats": sorted({path.suffix.lstrip(".").lower() for path in paths}),
                }
            )
        datasets.append(
            {
                "id": "all",
                "label": "all packaged datasets",
                "kind": "packaged",
                "file_count": sum(len(paths) for paths in groups.values()),
                "formats": ["flac"],
            }
        )
    # Uploaded datasets are owned by the API layer; imported lazily so the
    # experiments package stays usable without the api extra installed.
    try:
        from taf.api.uploads import upload_registry

        for uploaded in upload_registry.list():
            datasets.append(
                {
                    "id": f"{UPLOAD_DATASET_PREFIX}{uploaded.id}",
                    "label": uploaded.name,
                    "kind": "uploaded",
                    "file_count": len(uploaded.files),
                    "formats": sorted({path.suffix.lstrip(".").lower() for path in uploaded.files}),
                }
            )
    except Exception:  # pragma: no cover - api extra not installed
        pass
    return datasets


def dataset_exists(dataset_id: str | None, dataset_path: str | None) -> bool:
    if dataset_path is not None:
        return Path(dataset_path).is_dir()
    if dataset_id is None:
        return False
    return any(entry["id"] == dataset_id for entry in list_datasets())


__all__ = [
    "AttackParameter",
    "AttackSpec",
    "LONG_INPUT_METHODS",
    "TENSORFLOW_METHODS",
    "TENSORFLOW_METRICS",
    "dataset_exists",
    "list_attacks",
    "list_datasets",
    "list_methods",
    "list_metrics",
    "method_descriptions",
    "tensorflow_available",
]
