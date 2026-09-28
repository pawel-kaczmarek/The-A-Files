"""Reusable experiment execution engine.

Wraps the existing async evaluation engine (``taf.evaluation``) with the
normalized experiment schema: it validates configs, loads datasets, runs
encode/decode/attacks/metrics with per-row error isolation, and produces
normalized result rows plus a scenario-specific summary.
"""

from __future__ import annotations

import asyncio
import secrets
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from loguru import logger

from taf.experiments import registry
from taf.experiments.analysis import min_blocks_for_significance
from taf.experiments.provenance import build_manifest
from taf.experiments.results import ExperimentResultRow, normalize_row
from taf.experiments.schema import (
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
    #: Versions, input digests and resolved seed (``taf.experiments.provenance``).
    manifest: dict[str, Any] = field(default_factory=dict)
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


def _resolved_attacks(config: ExperimentConfig) -> list[str]:
    """Attack specifications for a run, expanding a named preset.

    A preset is resolved against the sample rate of the dataset so that the
    cutoffs and resampling targets it contains are valid for the material:
    a suite built for 44.1 kHz would otherwise ask for filters above the
    Nyquist frequency of 16 kHz speech. Explicitly listed attacks are kept
    and come first.
    """
    from taf.experiments.sweeps import attack_sweep_specs

    specs = list(config.attacks)
    for spec, _ in attack_sweep_specs(config.attack_sweep):
        if spec not in specs:
            specs.append(spec)
    if not config.attack_preset:
        return specs

    from taf.attacks.presets import benchmark_suite

    sample_rate = _dataset_sample_rate(config)
    for spec in benchmark_suite(config.attack_preset, sample_rate):
        if spec not in specs:
            specs.append(spec)
    return specs


def _dataset_sample_rate(config: ExperimentConfig) -> int:
    """Sample rate of the first file in the dataset, for preset resolution."""
    try:
        files = load_dataset_files(config)
    except Exception:  # noqa: BLE001 - dataset problems are reported elsewhere
        return 16000

    for path in files:
        try:
            return int(path.samplerate)
        except Exception:  # noqa: BLE001 - unreadable files are reported per row
            continue
    return 16000


#: Dataset-id prefix -> function returning the directory of that dataset.
_DATASET_RESOLVERS: dict[str, Callable[[str], Path | None]] = {}
_DATASET_MANIFEST_RESOLVERS: dict[str, Callable[[str], dict]] = {}


def register_dataset_resolver(prefix: str, resolver: Callable[[str], Path | None], manifest_resolver=None) -> None:
    """Let ``dataset_id="<prefix><key>"`` name a directory found by ``resolver(key)``.

    The platform registers its library this way, so the engine can run on
    library datasets without depending on the database.
    """
    _DATASET_RESOLVERS[prefix] = resolver
    if manifest_resolver is not None:
        _DATASET_MANIFEST_RESOLVERS[prefix] = manifest_resolver


def dataset_directory(config: ExperimentConfig) -> Path | None:
    """The directory a configuration reads from, if it is not a packaged set."""
    if config.dataset_path is not None:
        return Path(config.dataset_path)
    for prefix, resolver in _DATASET_RESOLVERS.items():
        if config.dataset_id and config.dataset_id.startswith(prefix):
            return resolver(config.dataset_id[len(prefix):])
    return None


def _is_external(config: ExperimentConfig) -> bool:
    return config.dataset_path is not None or bool(
        config.dataset_id and ":" in config.dataset_id
    )


def validate_config(config: ExperimentConfig) -> list[str]:
    """All scenario-independent + scenario-specific validation problems."""
    problems = validate_for_scenario(config)
    if _is_external(config):
        directory = dataset_directory(config)
        if directory is None or not directory.is_dir():
            problems.append(f"Dataset not found: {config.dataset_path or config.dataset_id!r}")
    elif not registry.dataset_exists(config.dataset_id, None):
        problems.append(f"Dataset not found: {config.dataset_id!r}")
    return problems


def load_dataset_files(config: ExperimentConfig):
    """Load the audio files an experiment will run on (existing loaders reused)."""
    from taf.audio.io import load_audio
    from taf.evaluation.workflow import load_files, load_resource_files
    from taf.resources.paths import example_wav_path, packaged_dataset_audio_paths

    if _is_external(config):
        directory = dataset_directory(config)
        if directory is None:
            raise ValueError(f"Dataset not found: {config.dataset_id!r}")
        files = load_files(directory, recursive=True)
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

    from taf.experiments.audio_inputs import prepare_inputs, select_files

    directory = dataset_directory(config) if _is_external(config) else None
    manifest = None
    for prefix, resolver in _DATASET_MANIFEST_RESOLVERS.items():
        if not config.dataset_path and config.dataset_id and config.dataset_id.startswith(prefix):
            manifest = resolver(config.dataset_id[len(prefix):])
    return prepare_inputs(select_files(files, config, directory), config, directory, manifest)


def _dataset_file_count(config: ExperimentConfig) -> int:
    """File count for previews without loading audio into memory."""
    from taf.resources.paths import packaged_dataset_audio_paths

    if _is_external(config):
        from taf.evaluation.workflow import audio_file_paths

        directory = dataset_directory(config)
        names = audio_file_paths(directory, recursive=True) if directory and directory.is_dir() else []
    elif config.dataset_id == "example":
        names = ["example.wav"]
    else:
        with packaged_dataset_audio_paths() as groups:
            if config.dataset_id == "all":
                paths = groups["vctk"] + groups["librispeech"]
            else:
                paths = groups.get(config.dataset_id or "", [])
            names = list(paths)

    from taf.experiments.audio_inputs import select_files

    return len(select_files(names, config, dataset_directory(config) if _is_external(config) else None))


def preview_experiment(config: ExperimentConfig) -> ExperimentPlan:
    """Dry-run estimate: counts, warnings and unsupported selections."""
    warnings: list[PlanWarning] = [
        PlanWarning(code="validation", message=problem) for problem in validate_config(config)
    ]
    unsupported_methods: list[str] = []
    unsupported_metrics: list[str] = []

    tensorflow = registry.tensorflow_available()
    methods = config.resolved_methods()
    for method in methods:
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
        if metric == "VISQOL_METRIC":
            from taf.metrics.speech_quality.VisqolMetric import model_path

            try:
                model_path()
            except ImportError as error:
                unsupported_metrics.append(metric)
                warnings.append(PlanWarning(code="missing_dependency", message=str(error)))
        if metric in registry.TENSORFLOW_METRICS and not tensorflow:
            unsupported_metrics.append(metric)
            warnings.append(
                PlanWarning(
                    code="missing_dependency",
                    message=f"{metric} requires TensorFlow (install the 'ai' extra); it will be recorded as a metric error.",
                )
            )
    from taf.attacks.registry import parse_spec, resolve_name

    changing = {spec.name for spec in registry.list_attacks() if spec.changes_length_or_rate}
    for attack in _resolved_attacks(config):
        try:
            attack_name = resolve_name(parse_spec(attack)[0])
        except Exception:  # noqa: BLE001 - validation reports this separately
            attack_name = attack
        if attack_name in changing:
            warnings.append(
                PlanWarning(
                    code="attack_changes_signal",
                    message=(
                        f"Attack '{attack}' changes signal length or sample rate; decodes will "
                        "usually fail and metrics on those rows are recorded as errors."
                    ),
                )
            )

    try:
        file_count = _dataset_file_count(config)
    except ValueError as error:
        file_count = 0
        warnings.append(PlanWarning(code="invalid_selection", message=str(error)))
    if file_count == 0:
        warnings.append(PlanWarning(code="empty_dataset", message="No audio files matched this selection."))
    elif (
        len(methods) > 1
        and config.experiment_type != ExperimentType.DETECTABILITY
        and file_count < (needed := min_blocks_for_significance(len(methods)))
    ):
        warnings.append(
            PlanWarning(
                code="few_files",
                message=(
                    f"Only {file_count} file(s): methods are compared per file, and with "
                    f"{len(methods)} methods no Holm-corrected pairwise test can reach "
                    f"p < 0.05 with fewer than {needed} files."
                ),
            )
        )
    if config.max_workers > 1 and config.experiment_type == ExperimentType.METHOD_COMPARISON:
        warnings.append(
            PlanWarning(
                code="concurrent_timing",
                message=(
                    f"max_workers={config.max_workers}: trials share the CPU, so encode/decode "
                    "times are not comparable and speed is left out of the comparison. "
                    "Use max_workers=1 to measure it."
                ),
            )
        )

    attack_variants = 1 + len(dict.fromkeys(_resolved_attacks(config)))
    encode_operations = file_count * len(methods) * config.payload_variant_count() * config.repetitions
    estimated_rows = encode_operations * attack_variants
    if config.experiment_type == ExperimentType.DETECTABILITY:
        # One steganalysis per (method, payload), reported in the summary;
        # the run produces no per-trial rows and ignores metrics.
        attack_variants, encode_operations, estimated_rows = 1, 0, 0
        if config.metrics:
            warnings.append(
                PlanWarning(
                    code="ignored_selection",
                    message="Detectability does not compute quality metrics; the selection is ignored.",
                )
            )
    return ExperimentPlan(
        experiment_type=config.experiment_type,
        file_count=file_count,
        method_count=len(methods),
        payload_length_count=config.payload_variant_count(),
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

    experiment_id = config.experiment_id or uuid.uuid4().hex[:12]
    output_dir = (
        Path(config.output_directory)
        if config.output_directory
        else Path(tempfile.gettempdir()) / "taf-experiments" / experiment_id
    )
    return EvaluationConfig(
        # Catalogue names resolve packaged and plugin components alike.
        methods=config.resolved_methods(),
        metrics=list(config.metrics),
        target=DecodeTarget.DIRECT,
        output_dir=output_dir,
        messages=config.payload.messages(config.repetitions) if config.payload.kind != "random" else (),
        random_message_rates_bps=config.payload_rates_bps,
        random_messages_per_length=config.repetitions,
        random_messages=[
            # No per-length seed: the engine derives one from ``random_seed``
            # and the length, so messages of different lengths are independent
            # draws rather than prefixes of one another.
            RandomMessageSpec(length=length, count=config.repetitions)
            for length in config.payload_lengths
        ] if config.payload.kind == "random" and not config.payload_rates_bps else [],
        random_seed=config.random_seed,
        attacks=_resolved_attacks(config),
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
    if config.random_seed is None:
        # Draw the seed now and keep it in the config, so the exported
        # configuration reproduces this run instead of drawing another one.
        config = config.model_copy(update={"random_seed": secrets.randbits(32)})
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
        if not files:
            raise ValueError("No audio files matched this experiment.")
        # Export the actual subset so adding unrelated files to a corpus cannot
        # silently change a replay. The library's original study stays reusable.
        config = config.model_copy(update={
            "selected_files": [file.metadata["file_id"] for file in files],
            "selected_file_sha256": {file.metadata["file_id"]: file.metadata["sha256"] for file in files},
        })
        run.config = config
        if config.experiment_type == ExperimentType.DETECTABILITY:
            from taf.experiments.detectability import run_detectability

            run.manifest = await asyncio.to_thread(build_manifest, config, files, [])
            run.summary = await asyncio.to_thread(run_detectability, config, files)
        else:
            evaluation_config = build_evaluation_config(config)
            run.manifest = await asyncio.to_thread(
                build_manifest, config, files, list(evaluation_config.attacks)
            )
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
    "dataset_directory",
    "register_dataset_resolver",
    "build_evaluation_config",
    "load_dataset_files",
    "preview_experiment",
    "run_experiment",
    "run_experiment_async",
    "validate_config",
]
