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
    #: Physical phenomenon the attack models (module of its class).
    family: str = ""
    #: Default robustness-curve sweep, if the attack has one.
    sweep: dict[str, Any] | None = None
    stochastic: bool = False
    #: Whether severity levels ("mp3@strong") are defined for the attack.
    has_severity: bool = False


def list_methods() -> list[dict[str, Any]]:
    """Packaged methods and those registered by plugins, with their metadata."""
    from taf.methods.catalog import METHOD_METADATA, method_abbreviation
    from taf.models.types import MethodType
    from taf.plugins import is_packaged_method, method_parameters, method_sources

    rows: list[dict[str, Any]] = []
    for name, source in sorted(method_sources().items()):
        try:
            method = create_method_safely(name)
        except Exception:  # noqa: BLE001 - a plugin that cannot be built is still listed
            method = None
        metadata = METHOD_METADATA.get(MethodType[name]) if name in MethodType.__members__ else None
        description = _safe_method_description(method) if method is not None else ""
        rows.append(
            {
                "name": name,
                "class_name": getattr(source, "__name__", type(source).__name__),
                "description": description,
                "abbreviation": method_abbreviation(name, description),
                "packaged": is_packaged_method(name),
                "family": metadata.family if metadata else None,
                "purpose": metadata.purpose if metadata else None,
                "reference": metadata.reference if metadata else None,
                "year": metadata.year if metadata else None,
                "doi": metadata.doi if metadata else None,
                "strength_parameter": metadata.strength_parameter if metadata else None,
                "parameters": method_parameters(name),
                "requires_tensorflow": name in TENSORFLOW_METHODS,
                "needs_long_input": name in LONG_INPUT_METHODS,
            }
        )
    return rows


def create_method_safely(name: str):
    from taf.plugins import create_method

    return create_method(name, _DEFAULT_SAMPLE_RATE)


def list_metrics() -> list[dict[str, Any]]:
    """Packaged metrics and those registered by plugins, with their direction."""
    from taf.metrics.catalog import METRIC_METADATA
    from taf.models.types import MetricType
    from taf.plugins import metric_factories

    rows: list[dict[str, Any]] = []
    for name, create in sorted(metric_factories().items()):
        metric = create()
        metadata = METRIC_METADATA.get(MetricType[name]) if name in MetricType.__members__ else None
        rows.append(
            {
                "name": name,
                "label": metric.name(),
                "class_name": metric.__class__.__name__,
                "category": _metric_category(metric.__class__.__module__),
                "packaged": name in MetricType.__members__,
                "requires_tensorflow": name in TENSORFLOW_METRICS,
                # True: higher means closer to the original; False: lower does.
                "higher_is_better": metric.higher_is_better,
                # Named entries of a multi-valued result, reported separately.
                "components": list(metric.components),
                "abbreviation": metadata.abbreviation if metadata else name,
                "scale": metadata.scale if metadata else None,
                "reference": metadata.reference if metadata else None,
                "year": metadata.year if metadata else None,
                "intrusive": metadata.intrusive if metadata else True,
                # All packaged metrics compare original vs processed samples and
                # require both signals to share length/sample rate.
                "compares_original": True,
                "supports_attacked_audio": True,
            }
        )
    return rows


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

    from taf.attacks.presets import sweep_presets
    from taf.attacks.registry import ATTACK_FACTORIES, attack_class, available_attacks

    sweeps = sweep_presets(_DEFAULT_SAMPLE_RATE)
    specs: list[AttackSpec] = []
    for name in available_attacks():
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
                family=cls.__module__.rsplit(".", 1)[-1],
                sweep=sweeps.get(name),
                stochastic=any(parameter.name == "seed" for parameter in parameters),
                has_severity=_has_severity(name),
            )
        )
    return specs


def _has_severity(name: str) -> bool:
    from taf.attacks.base import Severity
    from taf.attacks.presets import severity_parameters

    try:
        severity_parameters(name, Severity.MILD, _DEFAULT_SAMPLE_RATE)
    except Exception:  # noqa: BLE001 - no preset for this attack
        return False
    return True


def list_designs() -> list[dict[str, Any]]:
    """Experimental designs (``ExperimentType``) with their scientific structure."""
    from taf.experiments.scenarios import describe_designs

    return describe_designs()


def attack_presets(sample_rate: int = _DEFAULT_SAMPLE_RATE) -> dict[str, Any]:
    """Benchmark suites and sweep ladders resolved for a sampling rate."""
    from taf.attacks.presets import PIPELINES, SUITES, benchmark_suite, sweep_presets

    return {
        "sample_rate": sample_rate,
        "suites": {name: benchmark_suite(name, sample_rate) for name in SUITES},
        "sweeps": sweep_presets(sample_rate),
        "pipelines": {name: list(stages) for name, stages in PIPELINES.items()},
    }


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
    return datasets


def dataset_exists(dataset_id: str | None, dataset_path: str | None) -> bool:
    if dataset_path is not None:
        return Path(dataset_path).is_dir()
    if dataset_id is None:
        return False
    return any(entry["id"] == dataset_id for entry in list_datasets())


__all__ = [
    "AttackParameter",
    "attack_presets",
    "list_designs",
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
