"""Reusable experiment execution engine.

Wraps the existing async evaluation engine (``taf.evaluation``) with the
normalized experiment schema: it validates configs, loads datasets, runs
encode/decode/attacks/metrics with per-row error isolation, and produces
normalized result rows plus a scenario-specific summary.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from loguru import logger

from taf.experiments import registry
from taf.experiments.results import ExperimentResultRow, normalize_row
from taf.experiments.schema import (
    UPLOAD_DATASET_PREFIX,
    ExperimentConfig,
    ExperimentPlan,
    ExperimentType,
    PlanWarning,
)
from taf.experiments.scenarios import summarize_for_scenario, validate_for_scenario

# Minimum sample count for methods flagged as needing long inputs (they
# reserve floor(len/8192) frames and need >= 8 usable frames).
_LONG_INPUT_MIN_SAMPLES = 8192 * 8


@dataclass
class ExperimentRun:
    """Outcome of one experiment execution."""

    config: ExperimentConfig
    status: str = "pending"
    rows: list[ExperimentResultRow] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def experiment_id(self) -> str:
        return self.config.experiment_id or ""

    def to_csv(self, path: str | Path) -> Path:
        from taf.experiments.csv_export import export_detailed_csv

        target = Path(path)
        target.write_text(export_detailed_csv(self.rows), encoding="utf-8")
        return target


def validate_config(config: ExperimentConfig) -> list[str]:
    """All scenario-independent + scenario-specific validation problems."""
    problems = validate_for_scenario(config)
    if not registry.dataset_exists(config.dataset_id, config.dataset_path):
        source = config.dataset_path or config.dataset_id
        problems.append(f"Dataset not found: {source!r}")
    return problems


def load_dataset_files(config: ExperimentConfig):
    """Load the audio files an experiment will run on (existing loaders reused)."""
    from taf.audio.io import load_audio
    from taf.evaluation.workflow import load_files, load_resource_files
    from taf.resources.paths import example_wav_path, packaged_dataset_audio_paths

    if config.dataset_path is not None:
        files = load_files(config.dataset_path)
    elif config.dataset_id is not None and config.dataset_id.startswith(UPLOAD_DATASET_PREFIX):
        from taf.api.uploads import upload_registry

        upload_id = config.dataset_id[len(UPLOAD_DATASET_PREFIX):]
        uploaded = upload_registry.get(upload_id)
        if uploaded is None:
            raise ValueError(f"Uploaded dataset not found: {upload_id}")
        files = load_resource_files(uploaded.files)
    elif config.dataset_id == "example":
        with example_wav_path() as path:
            files = [load_audio(path)]
    else:
        with packaged_dataset_audio_paths() as groups:
            if config.dataset_id == "all":
                paths = groups["vctk"] + groups["librispeech"]
            else:
                paths = groups[config.dataset_id or ""]
            files = load_resource_files(paths)

    if config.selected_files:
        wanted = set(config.selected_files)
        files = [f for f in files if Path(f.path).name in wanted]
    if config.file_limit is not None:
        files = files[: config.file_limit]
    return files


def _dataset_file_count(config: ExperimentConfig) -> int:
    """File count for previews without loading audio into memory."""
    from taf.resources.paths import packaged_dataset_audio_paths

    if config.dataset_path is not None:
        directory = Path(config.dataset_path)
        names = [
            p.name
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
        ] if directory.is_dir() else []
    elif config.dataset_id is not None and config.dataset_id.startswith(UPLOAD_DATASET_PREFIX):
        from taf.api.uploads import upload_registry

        uploaded = upload_registry.get(config.dataset_id[len(UPLOAD_DATASET_PREFIX):])
        names = [p.name for p in uploaded.files] if uploaded else []
    elif config.dataset_id == "example":
        names = ["example.wav"]
    else:
        with packaged_dataset_audio_paths() as groups:
            if config.dataset_id == "all":
                paths = groups["vctk"] + groups["librispeech"]
            else:
                paths = groups.get(config.dataset_id or "", [])
            names = [p.name for p in paths]

    if config.selected_files:
        wanted = set(config.selected_files)
        names = [name for name in names if name in wanted]
    if config.file_limit is not None:
        names = names[: config.file_limit]
    return len(names)


def preview_experiment(config: ExperimentConfig) -> ExperimentPlan:
    """Dry-run estimate: counts, warnings and unsupported selections."""
    warnings: list[PlanWarning] = [
        PlanWarning(code="validation", message=problem) for problem in validate_config(config)
    ]
    unsupported_methods: list[str] = []
    unsupported_metrics: list[str] = []

    tensorflow = registry.tensorflow_available()
    for method in config.methods:
        if method in registry.TENSORFLOW_METHODS and not tensorflow:
            unsupported_methods.append(method)
            warnings.append(
                PlanWarning(
                    code="missing_dependency",
                    message=f"{method} requires TensorFlow (install the 'ai' extra); its rows will fail.",
                )
            )
        if method in registry.LONG_INPUT_METHODS:
            warnings.append(
                PlanWarning(
                    code="long_input",
                    message=(
                        f"{method} needs inputs of at least ~{_LONG_INPUT_MIN_SAMPLES} samples; "
                        "short files will produce failed rows."
                    ),
                )
            )
    for metric in config.metrics:
        if metric in registry.TENSORFLOW_METRICS and not tensorflow:
            unsupported_metrics.append(metric)
            warnings.append(
                PlanWarning(
                    code="missing_dependency",
                    message=f"{metric} requires TensorFlow (install the 'ai' extra); it will be recorded as a metric error.",
                )
            )
    changing = {spec.name for spec in registry.list_attacks() if spec.changes_length_or_rate}
    for attack in config.attacks:
        if attack in changing:
            warnings.append(
                PlanWarning(
                    code="attack_changes_signal",
                    message=(
                        f"Attack '{attack}' changes signal length or sample rate; decodes will "
                        "usually fail and metrics on those rows are recorded as errors."
                    ),
                )
            )

    file_count = _dataset_file_count(config)
    if file_count == 0:
        warnings.append(PlanWarning(code="empty_dataset", message="No audio files matched this selection."))

    attack_variants = 1 + len(dict.fromkeys(config.attacks))
    encode_operations = (
        file_count * len(config.methods) * len(config.payload_lengths) * config.repetitions
    )
    estimated_rows = encode_operations * attack_variants
    return ExperimentPlan(
        experiment_type=config.experiment_type,
        file_count=file_count,
        method_count=len(config.methods),
        payload_length_count=len(config.payload_lengths),
        repetitions=config.repetitions,
        attack_variant_count=attack_variants,
        metric_count=len(config.metrics),
        encode_operations=encode_operations,
        estimated_result_rows=estimated_rows,
        estimated_metric_calculations=estimated_rows * len(config.metrics),
        warnings=warnings,
        unsupported_methods=unsupported_methods,
        unsupported_metrics=unsupported_metrics,
    )


def build_evaluation_config(config: ExperimentConfig):
    """Translate the experiment config into the engine's EvaluationConfig."""
    from taf.audio.formats import DecodeTarget
    from taf.evaluation.config import EvaluationConfig
    from taf.evaluation.messages import RandomMessageSpec
    from taf.models.types import MethodType, MetricType

    experiment_id = config.experiment_id or uuid.uuid4().hex[:12]
    output_dir = (
        Path(config.output_directory)
        if config.output_directory
        else Path(tempfile.gettempdir()) / "taf-experiments" / experiment_id
    )
    return EvaluationConfig(
        methods=[MethodType[name] for name in config.methods],
        metrics=[MetricType[name] for name in config.metrics],
        target=DecodeTarget.DIRECT,
        output_dir=output_dir,
        random_messages=[
            RandomMessageSpec(length=length, count=config.repetitions, seed=config.random_seed)
            for length in config.payload_lengths
        ],
        random_seed=config.random_seed,
        attacks=list(config.attacks),
        keep_files=config.save_encoded_audio,
        max_workers=config.max_workers,
    )


async def run_experiment_async(
    config: ExperimentConfig,
    on_row: Callable[[ExperimentResultRow], None] | None = None,
) -> ExperimentRun:
    """Execute an experiment; per-row failures are recorded, never raised."""
    from taf.evaluation.workflow import evaluate_files_async

    if config.experiment_id is None:
        config = config.model_copy(update={"experiment_id": uuid.uuid4().hex[:12]})
    run = ExperimentRun(config=config, started_at=datetime.now(timezone.utc), status="running")

    problems = validate_config(config)
    if problems:
        run.status = "failed"
        run.error = "; ".join(problems)
        run.finished_at = datetime.now(timezone.utc)
        return run

    method_names = registry.method_descriptions()

    def handle_row(engine_row) -> None:
        normalized = normalize_row(
            engine_row,
            experiment_id=config.experiment_id or "",
            experiment_type=config.experiment_type.value,
            dataset_id=config.dataset_id or config.dataset_path,
            method_descriptions=method_names,
        )
        run.rows.append(normalized)
        if on_row is not None:
            on_row(normalized)

    try:
        files = await asyncio.to_thread(load_dataset_files, config)
        evaluation_config = build_evaluation_config(config)
        await evaluate_files_async(files, evaluation_config, on_row=handle_row)
        run.summary = summarize_for_scenario(run.rows, config)
        run.status = "completed"
    except Exception as error:  # dataset/setup level failure
        logger.exception("Experiment {} failed: {}", config.experiment_id, error)
        run.status = "failed"
        run.error = str(error)
    finally:
        run.finished_at = datetime.now(timezone.utc)
    return run


def run_experiment(
    config: ExperimentConfig,
    on_row: Callable[[ExperimentResultRow], None] | None = None,
) -> ExperimentRun:
    """Synchronous convenience wrapper for scripts and notebooks."""
    return asyncio.run(run_experiment_async(config, on_row=on_row))


__all__ = [
    "ExperimentRun",
    "build_evaluation_config",
    "load_dataset_files",
    "preview_experiment",
    "run_experiment",
    "run_experiment_async",
    "validate_config",
]
