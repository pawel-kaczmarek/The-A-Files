"""Discovery of methods, metrics, attacks and datasets for experiments.

Thin wrappers over the existing factories so that scripts, the API
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
    "awgn": "Additive white Gaussian noise at a target SNR.",
    "pink_noise": "Additive 1/f noise at a target SNR.",
    "impulse_noise": "Sparse high-amplitude impulses at a target SNR.",
    "codec": "Round trip through a real lossy encoder.",
    "mp3": "MP3 encode/decode round trip (libmp3lame).",
    "aac": "AAC encode/decode round trip.",
    "opus": "Opus encode/decode round trip (libopus).",
    "vorbis": "Vorbis encode/decode round trip (libvorbis).",
    "low_pass": "Butterworth low-pass filter.",
    "high_pass": "Butterworth high-pass filter.",
    "band_pass": "Butterworth band-pass filter.",
    "notch": "IIR notch removing a narrow band.",
    "smoothing": "Moving-average (boxcar FIR) smoothing.",
    "resample": "Sample-rate round trip through an intermediate rate.",
    "clock_drift": "Playback/capture clock mismatch in parts per million.",
    "bit_depth": "Uniform PCM requantisation to a lower bit depth.",
    "gain": "Amplitude scaling specified in decibels.",
    "clipping": "Hard clipping at an amplitude, peak or percentile threshold.",
    "compression_dynamic": "Static dynamic-range compression with make-up gain.",
    "time_shift": "Displacement along the time axis.",
    "crop": "Removal of a contiguous piece of the signal.",
    "dropout": "Zeroed runs of samples, preserving length.",
    "sample_jitter": "Insertion or deletion of short sample runs.",
    "time_stretch": "Duration change at constant pitch (phase vocoder).",
    "speed": "Playback speed change; duration and pitch move together.",
    "pitch_shift": "Pitch change at constant duration.",
    "echo": "Single delayed copy: y[n] = x[n] + a*x[n-D].",
    "reverb": "Convolution with a synthetic room impulse response.",
    "acoustic_channel": "Simulated loudspeaker, room, microphone path.",
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
    from taf.methods.factory import SteganographyMethodFactory

    methods = SteganographyMethodFactory._all_methods(_DEFAULT_SAMPLE_RATE)
    return [
        {
            "name": method_type.name,
            "class_name": method.__class__.__name__,
            "description": _safe_method_description(method),
            "requires_tensorflow": method_type.name in TENSORFLOW_METHODS,
            "needs_long_input": method_type.name in LONG_INPUT_METHODS,
        }
        for method_type, method in sorted(methods.items(), key=lambda item: item[0].name)
    ]


def list_metrics() -> list[dict[str, Any]]:
    from taf.metrics.factory import MetricFactory

    metrics = MetricFactory._all_methods()
    return [
        {
            "name": metric_type.name,
            "class_name": metric.__class__.__name__,
            "category": _metric_category(metric.__class__.__module__),
            "requires_tensorflow": metric_type.name in TENSORFLOW_METRICS,
            # All packaged metrics compare original vs processed samples and
            # require both signals to share length/sample rate.
            "compares_original": True,
            "supports_attacked_audio": True,
        }
        for metric_type, metric in sorted(metrics.items(), key=lambda item: item[0].name)
    ]


def _safe_method_description(method: object) -> str:
    type_function = getattr(method, "type", None)
    if not callable(type_function):
        return ""
    try:
        return str(type_function())
    except Exception:
        return ""


def _metric_category(module_name: str) -> str:
    parts = module_name.split(".")
    if "ai_based" in parts:
        return "ai_based"
    if "speech_intelligibility" in parts:
        return "speech_intelligibility"
    if "speech_quality" in parts:
        return "speech_quality"
    if "speech_reverberation" in parts:
        return "speech_reverberation"
    return "unknown"


def list_attacks() -> list[AttackSpec]:
    """Attack specifications taken from the attack registry.

    The parameters come from each attack's dataclass fields, so the catalogue
    always matches what the attack actually accepts. This replaced reflection
    over ``CorruptedWavFile`` methods, which also picked up helpers such as
    ``apply()`` and could not report defaults for parameters without one.
    """
    from dataclasses import MISSING, fields

    from taf.attacks.registry import ATTACK_CLASSES, ATTACK_FACTORIES, attack_class

    specs: list[AttackSpec] = []
    for name in sorted(set(ATTACK_CLASSES) | set(ATTACK_FACTORIES)):
        cls = attack_class(name)
        parameters = [
            AttackParameter(
                name=item.name,
                default=None if item.default is MISSING else item.default,
            )
            for item in fields(cls)
            if not (name in ATTACK_FACTORIES and item.name == "codec")
        ]
        specs.append(
            AttackSpec(
                name=name,
                class_name=cls.__name__,
                description=_ATTACK_DESCRIPTIONS.get(name, ""),
                parameters=parameters,
                changes_length_or_rate=bool(getattr(cls, "changes_length_or_rate", False)),
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
