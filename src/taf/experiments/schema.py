"""Canonical experiment configuration schema shared by scripts, API and UI."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from taf.experiments.sweeps import ParameterSweep

MIN_PAYLOAD_BITS = 4
# High enough for a capacity sweep of the sample-domain methods, which carry
# thousands of bits in a few seconds of speech; 120 bits measured reliability
# at small payloads rather than capacity.
MAX_PAYLOAD_BITS = 8192

BUILTIN_DATASETS = ("example", "vctk", "librispeech", "all")
#: Datasets of the platform library (``taf.persistence``), resolved to a
#: directory by a resolver registered with ``taf.experiments.runner``.
LIBRARY_DATASET_PREFIX = "library:"


class ExperimentType(str, Enum):
    DATASET_BENCHMARK = "dataset_benchmark"
    ATTACK_ROBUSTNESS = "attack_robustness"
    PERCEPTUAL_QUALITY = "perceptual_quality"
    EMBEDDING_CAPACITY = "embedding_capacity"
    METHOD_COMPARISON = "method_comparison"
    RESEARCH_EXPERIMENT = "research_experiment"
    DETECTABILITY = "detectability"
    ROBUSTNESS_CURVE = "robustness_curve"
    TRADEOFF_CURVE = "tradeoff_curve"


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
    #: Expand a named benchmark suite into ``attacks`` ("quick", "standard"
    #: or "full"). The suite is resolved when the run starts, against the
    #: sample rate of the material, so its filter cutoffs and resampling
    #: targets are valid for the dataset at hand.
    attack_preset: str | None = None
    #: One attack parameter varied over ordered values (robustness curve).
    attack_sweep: ParameterSweep | None = None
    #: One method parameter varied over ordered values (trade-off curve); the
    #: methods it produces are added to ``methods``.
    method_sweep: ParameterSweep | None = None

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
        """Names or specifications with constructor parameters.

        ``"QIM_METHOD:step_scale=0.2"`` is the QIM method with a larger
        quantisation step; two settings of one method may appear side by side.
        """
        from taf.plugins import method_spec_problems

        problems = [f"{value}: {problem}" for value in values for problem in method_spec_problems(value)]
        if problems:
            raise ValueError(f"Invalid method(s): {problems}")
        if len(set(values)) != len(values):
            raise ValueError("Methods must be unique.")
        return values

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, values: list[str]) -> list[str]:
        from taf.plugins import metric_names

        known = set(metric_names())
        unknown = [value for value in values if value not in known]
        if unknown:
            raise ValueError(f"Unknown metric(s): {unknown}. Known: {sorted(known)}")
        return values

    @field_validator("attack_preset")
    @classmethod
    def _known_preset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from taf.attacks.presets import SUITES

        key = str(value).lower()
        if key not in SUITES:
            raise ValueError(f"Unknown attack preset {value!r}. Known: {sorted(SUITES)}")
        return key

    @field_validator("attacks")
    @classmethod
    def _known_attacks(cls, values: list[str]) -> list[str]:
        """Accept the full attack specification grammar.

        An entry may carry parameters (``"awgn:snr_db=20"``), a severity
        (``"mp3@strong"``) or name a pipeline, so it cannot be checked against
        a list of bare names.
        """
        from taf.attacks.registry import available_attacks, unknown_specs

        unknown = unknown_specs(values)
        if unknown:
            raise ValueError(
                f"Unknown attack(s): {unknown}. Known: {sorted(available_attacks())}"
            )
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
        if value is None or value in BUILTIN_DATASETS or value.startswith(LIBRARY_DATASET_PREFIX):
            return value
        raise ValueError(
            f"Unknown dataset id: {value!r}. Use one of {list(BUILTIN_DATASETS)} "
            f"or '{LIBRARY_DATASET_PREFIX}<id>' for a dataset of the platform library."
        )

    @model_validator(mode="after")
    def _dataset_source_present(self) -> "ExperimentConfig":
        if self.dataset_id is None and self.dataset_path is None:
            raise ValueError("Either dataset_id or dataset_path is required.")
        if not self.methods and self.method_sweep is None:
            raise ValueError("At least one method is required.")
        return self

    @model_validator(mode="after")
    def _valid_sweeps(self) -> "ExperimentConfig":
        from taf.experiments.sweeps import attack_sweep_problems, method_sweep_problems

        problems: list[str] = []
        if self.attack_sweep is not None:
            problems += [f"attack_sweep: {p}" for p in attack_sweep_problems(self.attack_sweep)]
        if self.method_sweep is not None:
            problems += [f"method_sweep: {p}" for p in method_sweep_problems(self.method_sweep)]
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def resolved_methods(self) -> list[str]:
        """``methods`` followed by the settings of the swept method, if any."""
        from taf.experiments.sweeps import method_sweep_specs

        specs = list(self.methods)
        for spec, _ in method_sweep_specs(self.method_sweep):
            if spec not in specs:
                specs.append(spec)
        return specs


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
    "ParameterSweep",
    "ExperimentConfig",
    "ExperimentPlan",
    "ExperimentType",
    "MAX_PAYLOAD_BITS",
    "MIN_PAYLOAD_BITS",
    "PlanWarning",
    "LIBRARY_DATASET_PREFIX",
]
