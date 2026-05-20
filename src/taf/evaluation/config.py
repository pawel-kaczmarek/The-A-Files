from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from taf.audio.formats import AudioFileFormat, DecodeTarget
from taf.evaluation.messages import EvaluationMessage, RandomMessageSpec
from taf.models.Metric import Metric
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.types import MethodType, MetricType


class FailurePolicy(str, Enum):
    RECORD = "record"
    RAISE = "raise"


@dataclass
class EvaluationConfig:
    methods: Sequence[MethodType | SteganographyMethod | Callable[[int], SteganographyMethod]] | None = None
    metrics: Sequence[MetricType | Metric | Callable[[], Metric]] | None = None
    target: DecodeTarget = DecodeTarget.DIRECT
    formats: Sequence[AudioFileFormat | str] = (AudioFileFormat.WAV,)
    output_dir: Path = Path("artifacts") / "evaluation"
    messages: Sequence[EvaluationMessage | Sequence[int]] = ()
    random_messages: Sequence[RandomMessageSpec] = ()
    random_message_lengths: Sequence[int] = ()
    random_messages_per_length: int = 1
    random_seed: int | None = None
    keep_files: bool = True
    overwrite: bool = True
    max_workers: int = 1
    codec_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_policy: FailurePolicy = FailurePolicy.RECORD

    @classmethod
    def from_yaml(cls, path) -> "EvaluationConfig":
        """Load an ``EvaluationConfig`` from a YAML file. PyYAML is imported lazily."""
        from taf.evaluation.yaml_loader import load_config

        return load_config(path)

    def to_yaml(self, path) -> Path:
        """Serialize this config to a YAML file. PyYAML is imported lazily."""
        from taf.evaluation.yaml_loader import dump_config

        return dump_config(self, path)


__all__ = ["FailurePolicy", "EvaluationConfig"]
