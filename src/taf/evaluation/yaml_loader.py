"""YAML <-> ``EvaluationConfig`` serialization.

The loader is deliberately tolerant of partial YAML: any key absent from the
file keeps the dataclass default. Strings are coerced into the right enum
(``MethodType``, ``MetricType``, ``DecodeTarget``, ``AudioFileFormat``,
``FailurePolicy``) so a config file never needs to import Python types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

from taf.audio.formats import AudioFileFormat, DecodeTarget
from taf.evaluation.config import EvaluationConfig, FailurePolicy
from taf.evaluation.messages import EvaluationMessage, RandomMessageSpec
from taf.models.types import MethodType, MetricType


def load_config(path: str | Path) -> EvaluationConfig:
    """Parse a YAML file and return an ``EvaluationConfig``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return EvaluationConfig()
    if not isinstance(raw, dict):
        raise ValueError(
            f"Top-level YAML must be a mapping (got {type(raw).__name__}) in {path}"
        )
    return config_from_mapping(raw)


def dump_config(config: EvaluationConfig, path: str | Path) -> Path:
    """Serialize ``config`` back to YAML at ``path``. Best-effort for enum-only entries."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(config_to_mapping(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out


def config_from_mapping(data: dict[str, Any]) -> EvaluationConfig:
    """Build an ``EvaluationConfig`` from a parsed-YAML mapping. Unknown keys are ignored."""
    kwargs: dict[str, Any] = {}

    if (v := data.get("methods")) is not None:
        kwargs["methods"] = tuple(_method(item) for item in v)
    if (v := data.get("metrics")) is not None:
        kwargs["metrics"] = tuple(_metric(item) for item in v)
    if (v := data.get("target")) is not None:
        kwargs["target"] = _target(v)
    if (v := data.get("formats")) is not None:
        kwargs["formats"] = tuple(_format(item) for item in v)
    if (v := data.get("output_dir")) is not None:
        kwargs["output_dir"] = Path(v)
    if (v := data.get("messages")) is not None:
        kwargs["messages"] = tuple(_message(item, idx) for idx, item in enumerate(v))
    if (v := data.get("random_messages")) is not None:
        kwargs["random_messages"] = tuple(_random_spec(item) for item in v)
    if (v := data.get("random_message_lengths")) is not None:
        kwargs["random_message_lengths"] = tuple(int(x) for x in v)
    if "random_messages_per_length" in data:
        kwargs["random_messages_per_length"] = int(data["random_messages_per_length"])
    if "random_seed" in data:
        kwargs["random_seed"] = data["random_seed"]
    if "keep_files" in data:
        kwargs["keep_files"] = bool(data["keep_files"])
    if "overwrite" in data:
        kwargs["overwrite"] = bool(data["overwrite"])
    if "max_workers" in data:
        kwargs["max_workers"] = int(data["max_workers"])
    if (v := data.get("codec_options")) is not None:
        kwargs["codec_options"] = {str(k): dict(opts) for k, opts in v.items()}
    if (v := data.get("failure_policy")) is not None:
        kwargs["failure_policy"] = _failure_policy(v)

    return EvaluationConfig(**kwargs)


def config_to_mapping(config: EvaluationConfig) -> dict[str, Any]:
    """Render an ``EvaluationConfig`` back into a YAML-friendly mapping."""
    return {
        "target": config.target.value,
        "formats": [_format_to_str(item) for item in config.formats],
        "output_dir": str(config.output_dir),
        "methods": _enum_names(config.methods),
        "metrics": _enum_names(config.metrics),
        "messages": [_message_to_dict(item) for item in config.messages]
        if config.messages
        else [],
        "random_messages": [_random_spec_to_dict(item) for item in config.random_messages]
        if config.random_messages
        else [],
        "random_message_lengths": list(config.random_message_lengths),
        "random_messages_per_length": config.random_messages_per_length,
        "random_seed": config.random_seed,
        "keep_files": config.keep_files,
        "overwrite": config.overwrite,
        "max_workers": config.max_workers,
        "codec_options": dict(config.codec_options),
        "failure_policy": config.failure_policy.value,
    }


# --------------------------- coercion helpers ---------------------------


def _method(value: Any) -> MethodType:
    if isinstance(value, MethodType):
        return value
    try:
        return MethodType[str(value).upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown steganography method: {value!r}") from exc


def _metric(value: Any) -> MetricType:
    if isinstance(value, MetricType):
        return value
    try:
        return MetricType[str(value).upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown metric: {value!r}") from exc


def _target(value: Any) -> DecodeTarget:
    if isinstance(value, DecodeTarget):
        return value
    try:
        return DecodeTarget(str(value).lower())
    except ValueError as exc:
        raise ValueError(f"Unknown decode target: {value!r}") from exc


def _format(value: Any) -> AudioFileFormat:
    if isinstance(value, AudioFileFormat):
        return value
    return AudioFileFormat.from_value(value)


def _failure_policy(value: Any) -> FailurePolicy:
    if isinstance(value, FailurePolicy):
        return value
    return FailurePolicy(str(value).lower())


def _message(value: Any, index: int) -> EvaluationMessage | Sequence[int]:
    if isinstance(value, EvaluationMessage):
        return value
    if isinstance(value, dict):
        bits = tuple(int(b) for b in value["bits"])
        return EvaluationMessage(
            name=value.get("name", f"manual_{index:03d}"),
            bits=bits,
            source=value.get("source", "manual"),
            seed=value.get("seed"),
            index=value.get("index", index),
            metadata=dict(value.get("metadata", {})),
        )
    # raw bit list; let _materialize_messages name it
    return tuple(int(b) for b in value)


def _random_spec(value: Any) -> RandomMessageSpec:
    if isinstance(value, RandomMessageSpec):
        return value
    return RandomMessageSpec(
        length=int(value["length"]),
        count=int(value.get("count", 1)),
        seed=value.get("seed"),
        name_prefix=str(value.get("name_prefix", "random")),
    )


# --------------------------- dump helpers ---------------------------


def _enum_names(items: Any) -> list[str] | None:
    if not items:
        return None
    rendered: list[str] = []
    for item in items:
        if isinstance(item, (MethodType, MetricType)):
            rendered.append(item.name)
        else:
            # Non-enum entries (e.g. raw method instances or callables) — best-effort.
            rendered.append(getattr(item, "__name__", str(item)))
    return rendered


def _format_to_str(value: Any) -> str:
    if isinstance(value, (AudioFileFormat, DecodeTarget)):
        return value.value
    return str(value)


def _message_to_dict(message: EvaluationMessage) -> dict[str, Any]:
    return {
        "name": message.name,
        "bits": list(message.bits),
        "source": message.source,
        "seed": message.seed,
        "index": message.index,
        "metadata": dict(message.metadata),
    }


def _random_spec_to_dict(spec: RandomMessageSpec) -> dict[str, Any]:
    return {
        "length": spec.length,
        "count": spec.count,
        "seed": spec.seed,
        "name_prefix": spec.name_prefix,
    }


__all__ = [
    "load_config",
    "dump_config",
    "config_from_mapping",
    "config_to_mapping",
]
