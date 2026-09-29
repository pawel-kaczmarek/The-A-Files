"""Discovery of methods, metrics, attacks and datasets for experiments.

Thin wrappers over the existing factories so that scripts, the API
and the UI all see the same inventory. What each component is - its title,
descriptions, references and requirements - comes from the ``card`` declared
on its class (``taf.models.card``); nothing here restates it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_SAMPLE_RATE = 16000


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
    #: Physical phenomenon the attack models (``AttackCategory``).
    family: str = ""
    family_label: dict[str, str] = field(default_factory=dict)
    #: Default robustness-curve sweep, if the attack has one.
    sweep: dict[str, Any] | None = None
    stochastic: bool = False
    #: Whether severity levels ("mp3@strong") are defined for the attack.
    has_severity: bool = False
    #: Catalogue fields taken from the attack's card (``card_fields``).
    card: dict[str, Any] = field(default_factory=dict)


def card_fields(card: Any) -> dict[str, Any]:
    """Catalogue fields every component shares, taken from its card.

    Texts are expanded to every locale. The first reference is also flattened
    into ``reference``/``year``/``doi`` for tables that show one citation.
    """
    first = card.references[0] if card.references else None
    return {
        "title": card.title.as_dict(),
        "summary": card.summary.as_dict(),
        "details": card.details.as_dict(),
        "abbreviation": card.abbreviation,
        "references": [reference.as_dict() for reference in card.references],
        "requires": list(card.requires),
        "extra": card.extra,
        "available": not card.missing_requirements(),
        "reference": first.citation if first else None,
        "year": first.year if first else None,
        "doi": first.doi if first else None,
    }


def list_methods() -> list[dict[str, Any]]:
    """Packaged methods and those registered by plugins, described by their cards.

    Rows are ordered by family (``METHOD_FAMILIES``, then unknown families),
    then by name, so clients can group them without a list of families.
    """
    from taf.models.card import METHOD_FAMILIES, METHOD_PURPOSES, group_label, group_order
    from taf.plugins import is_packaged_method, method_card, method_parameters, method_sources

    rows: list[dict[str, Any]] = []
    for name, source in method_sources().items():
        try:
            method = create_method_safely(name)
        except Exception:  # noqa: BLE001 - a plugin that cannot be built is still listed
            method = None
        card = method_card(name)
        rows.append(
            {
                "name": name,
                "class_name": getattr(source, "__name__", type(source).__name__),
                "description": _safe_method_description(method) if method is not None else "",
                "packaged": is_packaged_method(name),
                **card_fields(card),
                "family": card.family,
                "family_label": group_label(METHOD_FAMILIES, card.family).as_dict(),
                "purpose": card.purpose,
                "purpose_label": METHOD_PURPOSES[card.purpose].as_dict() if card.purpose in METHOD_PURPOSES else None,
                "strength_parameter": card.strength_parameter,
                "parameters": method_parameters(name),
                "requires_tensorflow": "tensorflow" in card.requires,
                "needs_long_input": card.needs_long_input,
            }
        )
    return sorted(rows, key=lambda row: (group_order(METHOD_FAMILIES, row["family"]), row["name"]))


def create_method_safely(name: str):
    from taf.plugins import create_method

    return create_method(name, _DEFAULT_SAMPLE_RATE)


def list_metrics() -> list[dict[str, Any]]:
    """Packaged metrics and those registered by plugins, described by their cards.

    Rows are ordered by category (``METRIC_CATEGORIES``), then by name.
    """
    from taf.models.card import METRIC_CATEGORIES, group_label, group_order
    from taf.models.types import MetricType
    from taf.plugins import metric_card, metric_factories

    rows: list[dict[str, Any]] = []
    for name, create in metric_factories().items():
        metric = create()
        card = metric_card(name)
        category = card.category or _metric_category(metric.__class__.__module__)
        rows.append(
            {
                "name": name,
                "label": metric.name(),
                "class_name": metric.__class__.__name__,
                "packaged": name in MetricType.__members__,
                **card_fields(card),
                "category": category,
                "category_label": group_label(METRIC_CATEGORIES, category).as_dict(),
                "requires_tensorflow": "tensorflow" in card.requires,
                # True: higher means closer to the original; False: lower does.
                "higher_is_better": metric.higher_is_better,
                # Named entries of a multi-valued result, reported separately.
                "components": list(metric.components),
                "scale": card.scale,
                "intrusive": card.intrusive,
                "domain": card.domain,
                # All packaged metrics compare original vs processed samples and
                # require both signals to share length/sample rate.
                "compares_original": True,
                "supports_attacked_audio": True,
            }
        )
    return sorted(rows, key=lambda row: (group_order(METRIC_CATEGORIES, row["category"]), row["name"]))


def _safe_method_description(method: object) -> str:
    type_function = getattr(method, "type", None)
    if not callable(type_function):
        return ""
    try:
        return str(type_function())
    except Exception:
        return ""


def _metric_category(module_name: str) -> str:
    """Category of a metric without a card, from the package it lives in."""
    parts = module_name.split(".")
    if "ai_based" in parts:
        return "ai_based"
    if "speech_intelligibility" in parts:
        return "speech_intelligibility"
    if "speech_quality" in parts:
        return "speech_quality"
    if "speech_reverberation" in parts:
        return "speech_reverberation"
    return "other"


def list_attacks() -> list[AttackSpec]:
    """Attack specifications taken from the attack registry and their cards.

    The parameters come from each attack's dataclass fields, so the catalogue
    always matches what the attack actually accepts. This replaced reflection
    over ``CorruptedWavFile`` methods, which also picked up helpers such as
    ``apply()`` and could not report defaults for parameters without one.
    Specifications are ordered by family (``AttackCategory``), then by name.
    """
    from dataclasses import MISSING, fields

    from taf.attacks.base import AttackCategory
    from taf.attacks.presets import sweep_presets
    from taf.attacks.registry import ATTACK_FACTORIES, attack_card, attack_class, available_attacks, create
    from taf.models.card import ATTACK_FAMILIES, group_label, group_order

    sweeps = sweep_presets(_DEFAULT_SAMPLE_RATE)
    specs: list[AttackSpec] = []
    for name in available_attacks():
        cls = attack_class(name)
        card = attack_card(name)
        # A shortcut may fix defaults of its own (``vorbis`` lowers the bitrate).
        shortcut = create(name) if name in ATTACK_FACTORIES else None
        parameters = [
            AttackParameter(
                name=item.name,
                default=getattr(shortcut, item.name) if shortcut is not None
                else None if item.default is MISSING else item.default,
            )
            for item in fields(cls)
            if not (name in ATTACK_FACTORIES and item.name == "codec")
        ]
        category = getattr(cls, "category", None)
        family = category.value if isinstance(category, AttackCategory) else cls.__module__.rsplit(".", 1)[-1]
        specs.append(
            AttackSpec(
                name=name,
                class_name=cls.__name__,
                description=card.summary.en,
                parameters=parameters,
                changes_length_or_rate=bool(getattr(cls, "changes_length_or_rate", False)),
                family=family,
                family_label=group_label(ATTACK_FAMILIES, family).as_dict(),
                sweep=sweeps.get(name),
                stochastic=any(parameter.name == "seed" for parameter in parameters),
                has_severity=_has_severity(name),
                card=card_fields(card),
            )
        )
    return sorted(specs, key=lambda spec: (group_order(ATTACK_FAMILIES, spec.family), spec.name))


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
    "card_fields",
    "list_designs",
    "AttackSpec",
    "dataset_exists",
    "list_attacks",
    "list_datasets",
    "list_methods",
    "list_metrics",
    "method_descriptions",
    "tensorflow_available",
]
