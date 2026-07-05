"""Canonical experiment configuration schema shared by scripts, API and UI."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

MIN_PAYLOAD_BITS = 4
MAX_PAYLOAD_BITS = 120

BUILTIN_DATASETS = ("example", "vctk", "librispeech", "all")
UPLOAD_DATASET_PREFIX = "upload:"


class ExperimentType(str, Enum):
    DATASET_BENCHMARK = "dataset_benchmark"
    ATTACK_ROBUSTNESS = "attack_robustness"
    PERCEPTUAL_QUALITY = "perceptual_quality"
    EMBEDDING_CAPACITY = "embedding_capacity"
    METHOD_COMPARISON = "method_comparison"
    RESEARCH_EXPERIMENT = "research_experiment"


class ExperimentConfig(BaseModel):
    """One normalized configuration model for every experiment type.

    Scenario-specific requirements (e.g. attacks mandatory for attack
    robustness) are enforced by ``taf.experiments.scenarios``; this model
    validates everything that is scenario-independent.
    """

    experiment_id: str | None = None
    experiment_type: ExperimentType
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None

    # Data source: a known dataset id (builtin or "upload:<id>") and/or a
    # local directory path readable by the backend process.
    dataset_id: str | None = None
    dataset_path: str | None = None
    file_limit: int | None = Field(default=None, ge=1)
    selected_files: list[str] = Field(default_factory=list)

    methods: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    attacks: list[str] = Field(default_factory=list)

    payload_lengths: list[int] = Field(default_factory=lambda: [16])
    repetitions: int = Field(default=1, ge=1, le=50)
    random_seed: int | None = None

    output_directory: str | None = None
    save_encoded_audio: bool = False
    save_intermediate_results: bool = True

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    notes: str | None = None

    # Scenario knobs (thresholds for capacity, weights for comparison, ...).
    advanced_options: dict[str, Any] = Field(default_factory=dict)

    max_workers: int = Field(default=2, ge=1, le=16)

    @field_validator("methods")
    @classmethod
    def _known_methods(cls, values: list[str]) -> list[str]:
        from taf.models.types import MethodType

        known = {member.name for member in MethodType}
        unknown = [value for value in values if value not in known]
        if unknown:
            raise ValueError(f"Unknown method(s): {unknown}. Known: {sorted(known)}")
        return values

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, values: list[str]) -> list[str]:
        from taf.models.types import MetricType

        known = {member.name for member in MetricType}
        unknown = [value for value in values if value not in known]
        if unknown:
            raise ValueError(f"Unknown metric(s): {unknown}. Known: {sorted(known)}")
        return values

    @field_validator("attacks")
    @classmethod
    def _known_attacks(cls, values: list[str]) -> list[str]:
        from taf.evaluation.workflow import available_attack_names

        known = set(available_attack_names())
        unknown = [value for value in values if value not in known]
        if unknown:
            raise ValueError(f"Unknown attack(s): {unknown}. Known: {sorted(known)}")
        return values

    @field_validator("payload_lengths")
    @classmethod
    def _valid_payload_lengths(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("At least one payload length is required.")
        bad = [v for v in values if v < MIN_PAYLOAD_BITS or v > MAX_PAYLOAD_BITS]
        if bad:
            raise ValueError(
                f"Payload lengths must be between {MIN_PAYLOAD_BITS} and {MAX_PAYLOAD_BITS} bits; got {bad}."
            )
        if len(set(values)) != len(values):
            raise ValueError("Payload lengths must be unique.")
        return values

    @field_validator("dataset_id")
    @classmethod
    def _known_dataset(cls, value: str | None) -> str | None:
        if value is None or value in BUILTIN_DATASETS or value.startswith(UPLOAD_DATASET_PREFIX):
            return value
        raise ValueError(
            f"Unknown dataset id: {value!r}. Use one of {list(BUILTIN_DATASETS)} "
            f"or '{UPLOAD_DATASET_PREFIX}<id>' for an uploaded dataset."
        )

    @model_validator(mode="after")
    def _dataset_source_present(self) -> "ExperimentConfig":
        if self.dataset_id is None and self.dataset_path is None:
            raise ValueError("Either dataset_id or dataset_path is required.")
        if not self.methods:
            raise ValueError("At least one method is required.")
        return self


class PlanWarning(BaseModel):
    code: str
    message: str


class ExperimentPlan(BaseModel):
    """Dry-run estimate returned by the preview endpoint."""

    experiment_type: ExperimentType
    file_count: int
    method_count: int
    payload_length_count: int
    repetitions: int
    attack_variant_count: int
    metric_count: int
    encode_operations: int
    estimated_result_rows: int
    estimated_metric_calculations: int
    warnings: list[PlanWarning] = Field(default_factory=list)
    unsupported_methods: list[str] = Field(default_factory=list)
    unsupported_metrics: list[str] = Field(default_factory=list)
    unsupported_attacks: list[str] = Field(default_factory=list)


__all__ = [
    "BUILTIN_DATASETS",
    "ExperimentConfig",
    "ExperimentPlan",
    "ExperimentType",
    "MAX_PAYLOAD_BITS",
    "MIN_PAYLOAD_BITS",
    "PlanWarning",
    "UPLOAD_DATASET_PREFIX",
]
