from __future__ import annotations

import asyncio
import copy
import hashlib
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from taf.audio.formats import AudioFileFormat, DecodeTarget
from taf.audio.io import load_audio, save_audio
from taf.evaluation.config import EvaluationConfig, FailurePolicy
from taf.evaluation.messages import EvaluationMessage, RandomMessageSpec
from taf.evaluation.result import EvaluationResult, EvaluationRow, FailureKind
from taf.evaluation.seeding import attack_seed, message_seed
from taf.models.Metric import Metric
from taf.models.errors import CapacityError
from taf.models.SteganographyMethod import SteganographyMethod
from taf.models.WavFile import WavFile
from taf.models.types import MethodType, MetricType


@dataclass(frozen=True)
class _MethodSpec:
    label: str
    create: Callable[[int], SteganographyMethod]
    #: Catalogue name and constructor parameters, when the method was named.
    name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _MetricSpec:
    label: str
    create: Callable[[], Metric]


def audio_file_paths(path: str | Path, recursive: bool = False) -> list[Path]:
    """WAV and FLAC files of a directory, in name order."""
    directory = Path(path)
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        file_path
        for file_path in candidates
        if file_path.is_file()
        and file_path.suffix.lower() in {AudioFileFormat.FLAC.extension, AudioFileFormat.WAV.extension}
    )


def load_files(path: str | Path, recursive: bool = False) -> list[WavFile]:
    directory = Path(path)
    logger.debug("Scanning directory for audio files: {}", directory)
    files = audio_file_paths(directory, recursive=recursive)
    logger.debug("Loading {} audio file(s) from {}", len(files), directory)
    return [WavFile.load(file_path) for file_path in files]


def load_resource_files(paths: Iterable[Path]) -> list[WavFile]:
    paths_list = list(paths)
    logger.debug("Loading {} packaged-resource audio file(s)", len(paths_list))
    return [load_audio(path) for path in paths_list]


def evaluate_files(
    files: Iterable[WavFile],
    config: EvaluationConfig | None = None,
    on_row: Callable[[EvaluationRow], None] | None = None,
) -> EvaluationResult:
    return asyncio.run(evaluate_files_async(files, config, on_row=on_row))


async def evaluate_files_async(
    files: Iterable[WavFile],
    config: EvaluationConfig | None = None,
    on_row: Callable[[EvaluationRow], None] | None = None,
) -> EvaluationResult:
    files_list = list(files)
    resolved_config = config or EvaluationConfig()
    messages = _materialize_messages(resolved_config)
    method_specs = _method_specs(resolved_config)
    metric_specs = _metric_specs(resolved_config)
    targets = _resolve_targets(resolved_config)
    attack_variants = _resolve_attack_variants(resolved_config)
    semaphore = asyncio.Semaphore(max(1, resolved_config.max_workers))

    total_tasks = len(files_list) * len(method_specs) * len(messages)
    logger.info(
        "Evaluating {} file(s) x {} method(s) x {} message(s) x {} attack variant(s) -> {} task(s); "
        "target={} formats={} max_workers={} failure_policy={} output_dir={}",
        len(files_list),
        len(method_specs),
        len(messages),
        len(attack_variants),
        total_tasks,
        resolved_config.target.value,
        [_format_value(t) or _decode_mode(t) for t in targets],
        resolved_config.max_workers,
        resolved_config.failure_policy.value,
        resolved_config.output_dir,
    )
    logger.debug("Methods: {}", [spec.label for spec in method_specs])
    logger.debug("Metrics: {}", [spec.label for spec in metric_specs])
    logger.debug("Messages: {}", [(m.name, m.length) for m in messages])

    async def guarded_job(wav_file: WavFile, method_spec: _MethodSpec, message: EvaluationMessage):
        async with semaphore:
            job_rows = await asyncio.to_thread(
                _evaluate_file_method_message,
                wav_file,
                method_spec,
                message,
                metric_specs,
                targets,
                attack_variants,
                resolved_config,
            )
        if on_row is not None:
            for row in job_rows:
                on_row(row)
        return job_rows

    tasks = [
        guarded_job(wav_file, method_spec, message)
        for wav_file in files_list
        for method_spec in method_specs
        for message in messages
    ]
    start = time.perf_counter()
    row_groups = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    rows = [row for group in row_groups for row in group]
    logger.info(
        "Async evaluation finished in {:.2f}s — {} task(s) produced {} row(s)",
        elapsed,
        len(tasks),
        len(rows),
    )
    return EvaluationResult(messages={message.name: message for message in messages}, rows=rows)


def _evaluate_file_method_message(
    wav_file: WavFile,
    method_spec: _MethodSpec,
    message: EvaluationMessage,
    metric_specs: Sequence[_MetricSpec],
    targets: Sequence[AudioFileFormat | DecodeTarget],
    attack_variants: Sequence[str | None],
    config: EvaluationConfig,
) -> list[EvaluationRow]:
    rows = _encode_and_evaluate(
        wav_file, method_spec, message, metric_specs, targets, attack_variants, config
    )
    for row in rows:
        row.method_name = method_spec.name
        row.method_parameters = dict(method_spec.parameters)
    return rows


def _encode_and_evaluate(
    wav_file: WavFile,
    method_spec: _MethodSpec,
    message: EvaluationMessage,
    metric_specs: Sequence[_MetricSpec],
    targets: Sequence[AudioFileFormat | DecodeTarget],
    attack_variants: Sequence[str | None],
    config: EvaluationConfig,
) -> list[EvaluationRow]:
    method = method_spec.create(wav_file.samplerate)
    method_label = _method_label(method, method_spec)

    logger.debug(
        "Encoding | file={} | method={} | message={} | bits={}",
        wav_file.path,
        method_label,
        message.name,
        message.length,
    )

    encode_start = time.perf_counter()
    try:
        encoded_samples = method.encode(wav_file.samples.copy(), list(message.bits))
    except Exception as error:
        if isinstance(error, CapacityError):
            # Expected in a capacity sweep: the outcome of the trial, not a fault.
            failure_kind = FailureKind.OVER_CAPACITY
            logger.info(
                "Over capacity | file={} | method={} | message={} | {}",
                wav_file.path,
                method_label,
                message.name,
                error,
            )
        else:
            failure_kind = FailureKind.ENCODE_ERROR
            logger.exception(
                "Encode failed | file={} | method={} | message={} | error={}",
                wav_file.path,
                method_label,
                message.name,
                error,
            )
        if config.failure_policy == FailurePolicy.RAISE:
            raise
        return [
            _error_row(
                wav_file=wav_file,
                method=method_label,
                message=message,
                target=target,
                error=error,
                failure_kind=failure_kind,
                attack=attack,
                encode_time=time.perf_counter() - encode_start,
            )
            for target in targets
            for attack in attack_variants
        ]
    encode_elapsed = time.perf_counter() - encode_start
    logger.debug(
        "Encode done  | file={} | method={} | message={} | took={:.3f}s | out_samples={}",
        wav_file.path,
        method_label,
        message.name,
        encode_elapsed,
        len(encoded_samples),
    )

    # Extraction is blind: it runs on a fresh instance that has seen neither
    # the cover nor the message, so no state kept by encode() can help it.
    decoder = method_spec.create(wav_file.samplerate)
    encoded = WavFile(samplerate=wav_file.samplerate, samples=encoded_samples, path=wav_file.path)
    rows: list[EvaluationRow] = []
    for target in targets:
        rows.extend(
            _evaluate_target(
                wav_file,
                encoded,
                decoder,
                method_label,
                message,
                metric_specs,
                target,
                attack_variants,
                config,
                encode_elapsed,
            )
        )
    return rows


def _evaluate_target(
    original: WavFile,
    encoded: WavFile,
    decoder: SteganographyMethod,
    method_label: str,
    message: EvaluationMessage,
    metric_specs: Sequence[_MetricSpec],
    target: AudioFileFormat | DecodeTarget,
    attack_variants: Sequence[str | None],
    config: EvaluationConfig,
    encode_time: float | None = None,
) -> list[EvaluationRow]:
    """Every attack variant of one encoded message delivered through one target."""
    output_path = _output_path(original.path, method_label, message.name, target, config)
    target_label = _format_value(target) or _decode_mode(target)

    try:
        if isinstance(target, DecodeTarget):
            logger.debug(
                "Direct decode | file={} | method={} | message={} | target={}",
                original.path,
                method_label,
                message.name,
                target_label,
            )
            stego = encoded
            output_path = None
        else:
            if output_path.exists() and not config.overwrite:
                raise FileExistsError(f"Output file already exists: {output_path}")
            options = config.codec_options.get(target.value, {})
            logger.debug(
                "Saving encoded audio | path={} | format={} | options={}",
                output_path,
                target.value,
                options or "<defaults>",
            )
            saved_path = save_audio(encoded, output_path, target, options)
            stego = load_audio(saved_path)
            logger.debug(
                "Reloaded after save | path={} | samples={} | samplerate={}",
                saved_path,
                len(stego.samples),
                stego.samplerate,
            )
            if not config.keep_files:
                _remove_file(saved_path)
                logger.debug("Removed transient audio artifact: {}", saved_path)
                output_path = None
    except Exception as error:
        logger.exception(
            "Target preparation failed | file={} | method={} | message={} | target={} | error={}",
            original.path,
            method_label,
            message.name,
            target_label,
            error,
        )
        if config.failure_policy == FailurePolicy.RAISE:
            raise
        return [
            _error_row(
                original,
                method_label,
                message,
                target,
                error,
                output_path,
                failure_kind=FailureKind.IO_ERROR,
                attack=attack,
                encode_time=encode_time,
            )
            for attack in attack_variants
        ]

    # Imperceptibility and robustness are different measurements and are kept
    # apart. `metrics` always compares the cover with the stego signal, so it
    # answers "how much did embedding change the audio". It does not depend on
    # the attack, so it is computed once here and shared by every attack
    # variant, and it is kept even when extraction later fails. Attack damage
    # is reported separately, against the stego signal the attacker received,
    # so the two effects are never summed into one number.
    metrics, metric_errors = _calculate_metrics(
        original.samples, stego.samples, original.samplerate, metric_specs
    )
    return [
        _evaluate_attack_variant(
            original,
            stego,
            decoder,
            method_label,
            message,
            metric_specs,
            target,
            attack,
            config,
            encode_time,
            output_path,
            metrics,
            metric_errors,
        )
        for attack in attack_variants
    ]


def _evaluate_attack_variant(
    original: WavFile,
    stego: WavFile,
    decoder: SteganographyMethod,
    method_label: str,
    message: EvaluationMessage,
    metric_specs: Sequence[_MetricSpec],
    target: AudioFileFormat | DecodeTarget,
    attack: str | None,
    config: EvaluationConfig,
    encode_time: float | None,
    output_path: Path | None,
    metrics: dict[str, Any],
    metric_errors: dict[str, str],
) -> EvaluationRow:
    target_label = _format_value(target) or _decode_mode(target)
    stego_samples = stego.samples
    decode_samples = stego_samples
    attack_elapsed: float | None = None
    attack_metadata: dict[str, Any] = {}
    failure_kind = FailureKind.ATTACK_ERROR

    try:
        if attack is not None:
            from taf.attacks.registry import has_explicit_seed

            seed = None if has_explicit_seed(attack) else attack_seed(
                _base_seed(config), _file_key(original.path), _repetition(message), attack
            )
            logger.debug(
                "Applying attack | file={} | method={} | message={} | target={} | attack={} | seed={}",
                original.path,
                method_label,
                message.name,
                target_label,
                attack,
                seed if seed is not None else "<from specification>",
            )
            attack_start = time.perf_counter()
            decode_samples, _, attack_metadata = _apply_attack(
                decode_samples, stego.samplerate, attack, seed
            )
            attack_elapsed = time.perf_counter() - attack_start

        failure_kind = FailureKind.DECODE_ERROR
        decode_start = time.perf_counter()
        decoded_message = decoder.decode(decode_samples, message.length)
        decode_elapsed = time.perf_counter() - decode_start
    except Exception as error:
        logger.exception(
            "{} | file={} | method={} | message={} | target={} | attack={} | error={}",
            failure_kind,
            original.path,
            method_label,
            message.name,
            target_label,
            attack or "none",
            error,
        )
        if config.failure_policy == FailurePolicy.RAISE:
            raise
        return _error_row(
            original,
            method_label,
            message,
            target,
            error,
            output_path,
            failure_kind=failure_kind,
            attack=attack,
            encode_time=encode_time,
            attack_time=attack_elapsed,
            attack_parameters=attack_metadata,
            metrics=metrics,
            metric_errors=metric_errors,
        )

    success = bool(np.array_equal(list(message.bits), decoded_message))
    logger.debug(
        "Decode done  | file={} | method={} | message={} | target={} | attack={} | took={:.3f}s | success={}",
        original.path,
        method_label,
        message.name,
        target_label,
        attack or "none",
        decode_elapsed,
        success,
    )

    attack_metrics: dict[str, Any] = {}
    attack_metric_errors: dict[str, str] = {}
    if attack is not None:
        if len(decode_samples) == len(stego_samples):
            attack_metrics, attack_metric_errors = _calculate_metrics(
                stego_samples, decode_samples, original.samplerate, metric_specs
            )
        else:
            # Cropping, stretching and padding change the length, and the
            # packaged metrics all require two signals of equal length.
            attack_metric_errors = {
                "*": (
                    f"attack changed the signal length "
                    f"({len(stego_samples)} -> {len(decode_samples)} samples); "
                    "sample-aligned quality metrics do not apply"
                )
            }
    return EvaluationRow(
        input_path=original.path,
        method=method_label,
        message_name=message.name,
        message_length=message.length,
        decode_mode=_decode_mode(target),
        format=_format_value(target),
        success=success,
        metrics=metrics,
        metric_errors=metric_errors,
        attack_metrics=attack_metrics,
        attack_metric_errors=attack_metric_errors,
        output_path=output_path,
        decoded_message=list(decoded_message),
        repetition=message.index,
        is_lossy=_is_lossy(target),
        transformation_name=_transformation_name(target),
        codec_options=config.codec_options.get(_format_value(target) or "", {}),
        attack=attack,
        attack_parameters=attack_metadata,
        message_bits=list(message.bits),
        sample_rate=original.samplerate,
        duration_seconds=len(original.samples) / original.samplerate if original.samplerate else None,
        encode_time_seconds=encode_time,
        decode_time_seconds=decode_elapsed,
        attack_time_seconds=attack_elapsed,
    )


def _base_seed(config: EvaluationConfig) -> int:
    """The experiment seed; 0 keeps attack realisations reproducible without one."""
    return config.random_seed if config.random_seed is not None else 0


def _file_key(path: Path) -> str:
    # The file name rather than the full path, so a dataset moved to another
    # directory or machine reproduces the same attack realisations.
    return Path(path).name


def _repetition(message: EvaluationMessage) -> int:
    return message.index if message.index is not None else 0


def _calculate_metrics(
    original_samples: np.ndarray,
    processed_samples: np.ndarray,
    samplerate: int,
    metric_specs: Sequence[_MetricSpec],
) -> tuple[dict[str, Any], dict[str, str]]:
    metrics: dict[str, Any] = {}
    errors: dict[str, str] = {}
    logger.debug(
        "Computing {} metric(s) at samplerate={} Hz over {} samples",
        len(metric_specs),
        samplerate,
        len(original_samples),
    )
    for metric_spec in metric_specs:
        metric = metric_spec.create()
        name = metric.name()
        metric_start = time.perf_counter()
        try:
            value = metric.calculate(original_samples, processed_samples, samplerate, 0.03, 0.75)
        except Exception as error:
            errors[name] = str(error)
            logger.warning("Metric '{}' raised: {}", name, error)
            continue
        # Multi-valued metrics are split into named components here, where the
        # metric that knows their meaning is at hand; averaging them later
        # would mix unrelated quantities.
        try:
            metrics.update(metric.labelled_values(value))
        except (TypeError, ValueError) as error:
            errors[name] = f"non-numeric result: {error}"
            continue
        logger.debug(
            "Metric ok    | name={} | took={:.3f}s | value={}",
            name,
            time.perf_counter() - metric_start,
            value,
        )
    return metrics, errors


def _materialize_messages(config: EvaluationConfig) -> list[EvaluationMessage]:
    messages: list[EvaluationMessage] = []

    for index, message in enumerate(config.messages):
        if isinstance(message, EvaluationMessage):
            messages.append(_normalize_message(message))
        else:
            bits = _validate_bits(message)
            messages.append(EvaluationMessage(name=f"manual_{index:03d}", bits=tuple(bits), index=index))

    specs = list(config.random_messages)
    if config.random_message_lengths:
        for length in config.random_message_lengths:
            specs.append(
                RandomMessageSpec(
                    length=length,
                    count=config.random_messages_per_length,
                )
            )

    generated_seeds = _generated_seeds(specs, config.random_seed)
    for spec_index, spec in enumerate(specs):
        if spec.length < 0:
            raise ValueError("Random message length must be non-negative.")
        if spec.count < 1:
            raise ValueError("Random message count must be positive.")
        seed = spec.seed if spec.seed is not None else generated_seeds[spec_index]
        rng = np.random.default_rng(seed)
        for message_index in range(spec.count):
            bits = tuple(int(bit) for bit in rng.integers(0, 2, size=spec.length))
            messages.append(
                EvaluationMessage(
                    name=f"{spec.name_prefix}_{spec_index:03d}_len{spec.length}_{message_index:03d}",
                    bits=bits,
                    source="random",
                    seed=seed,
                    index=message_index,
                    metadata={"spec_index": spec_index},
                )
            )

    if not messages:
        default_spec = RandomMessageSpec(length=10, count=1, seed=config.random_seed)
        return _materialize_messages(
            EvaluationConfig(
                messages=(),
                random_messages=(default_spec,),
                random_seed=config.random_seed,
            )
        )

    _validate_unique_message_names(messages)
    return messages


def _method_specs(config: EvaluationConfig) -> list[_MethodSpec]:
    if config.methods is None:
        return [_method_type_spec(method_type) for method_type in MethodType]

    specs: list[_MethodSpec] = []
    for method in config.methods:
        if isinstance(method, MethodType):
            specs.append(_method_type_spec(method))
        elif isinstance(method, str):
            specs.append(_named_method_spec(method))
        elif isinstance(method, SteganographyMethod):
            label = method.type()
            specs.append(_MethodSpec(label=label, create=lambda sr, original=method: copy.deepcopy(original)))
        elif callable(method):
            specs.append(_MethodSpec(label=_callable_label(method), create=method))
        else:
            raise TypeError(f"Unsupported method config entry: {method!r}")
    return specs


def _metric_specs(config: EvaluationConfig) -> list[_MetricSpec]:
    if config.metrics is None:
        from taf.metrics.factory import MetricFactory

        return [_MetricSpec(label=metric.name(), create=lambda original=metric: copy.deepcopy(original)) for metric in MetricFactory.get_all()]

    specs: list[_MetricSpec] = []
    for metric in config.metrics:
        if isinstance(metric, MetricType):
            specs.append(_metric_type_spec(metric))
        elif isinstance(metric, str):
            specs.append(_named_metric_spec(metric))
        elif isinstance(metric, Metric):
            specs.append(_MetricSpec(label=metric.name(), create=lambda original=metric: copy.deepcopy(original)))
        elif callable(metric):
            specs.append(_MetricSpec(label=_callable_label(metric), create=metric))
        else:
            raise TypeError(f"Unsupported metric config entry: {metric!r}")
    return specs


def _method_type_spec(method_type: MethodType) -> _MethodSpec:
    def create(samplerate: int) -> SteganographyMethod:
        from taf.methods.factory import SteganographyMethodFactory

        method = SteganographyMethodFactory.get(samplerate, method_type)
        if method is None:
            raise ValueError(f"Unknown steganography method: {method_type}")
        return method

    return _MethodSpec(label=method_type.name, create=create, name=method_type.name)


def _metric_type_spec(metric_type: MetricType) -> _MetricSpec:
    def create() -> Metric:
        from taf.metrics.factory import MetricFactory

        metric = MetricFactory.get(metric_type)
        if metric is None:
            raise ValueError(f"Unknown metric: {metric_type}")
        return metric

    return _MetricSpec(label=metric_type.name, create=create)


def _named_method_spec(spec: str) -> _MethodSpec:
    """A method by catalogue name or specification (``"QIM_METHOD:step_scale=0.2"``)."""
    from taf.plugins import create_method, method_spec_problems, parse_method_spec

    problems = method_spec_problems(spec)
    if problems:
        raise ValueError(f"Invalid steganography method {spec!r}: {'; '.join(problems)}")
    name, parameters = parse_method_spec(spec)
    return _MethodSpec(
        label=spec,
        create=lambda samplerate: create_method(spec, samplerate),
        name=name,
        parameters=parameters,
    )


def _named_metric_spec(name: str) -> _MetricSpec:
    """A metric by catalogue name: packaged (``"SNR_METRIC"``) or plugin."""
    from taf.plugins import create_metric, metric_names

    if name not in metric_names():
        raise ValueError(f"Unknown metric: {name!r}")
    return _MetricSpec(label=name, create=lambda: create_metric(name))


def available_attack_names() -> list[str]:
    """Attack names the registry can build (sorted)."""
    from taf.attacks.registry import available_attacks

    return available_attacks()


def _resolve_attack_variants(config: EvaluationConfig) -> list[str | None]:
    """Each configured attack is evaluated as its own variant, next to a no-attack baseline.

    An entry may be a bare name (``"awgn"``), a parameterised specification
    (``"awgn:snr_db=20"``), a severity (``"mp3@strong"``) or a named pipeline
    (``"pipeline:name=voice_call"``). Specifications are validated here so a
    typo fails before any audio is processed.
    """
    if not config.attacks:
        return [None]

    from taf.attacks.registry import unknown_specs

    unknown = unknown_specs(config.attacks)
    if unknown:
        raise ValueError(
            f"Unknown attack(s): {unknown}. Known: {sorted(available_attack_names())}"
        )

    variants: list[str | None] = [None]
    for spec in config.attacks:
        if spec not in variants:
            variants.append(spec)
    return variants


def _apply_attack(
    samples: np.ndarray, samplerate: int, attack: str, seed: int | None = None
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Apply one attack specification, returning the audio and its metadata.

    The metadata is what makes a benchmark row reproducible: it carries the
    resolved parameters, the seed, the input and output rates and lengths, and
    any clipping or length correction the attack performed. ``seed``, when
    given, replaces the seed of every random stage of the attack.
    """
    from taf.attacks.registry import build, reseed

    built = build(attack, sample_rate=samplerate)
    if seed is not None:
        built = reseed(built, seed)
    result = built.apply(np.asarray(samples), samplerate)
    return np.asarray(result.audio), result.sample_rate, dict(result.metadata)


def _resolve_targets(config: EvaluationConfig) -> list[AudioFileFormat | DecodeTarget]:
    """Translate the (target, formats) pair on a config into concrete iteration targets.

    DIRECT mode produces a single in-memory target and ignores ``formats``.
    FILES mode requires ``formats`` to be a non-empty list of concrete audio
    containers; each one becomes its own row per (file, method, message).
    """
    if config.target == DecodeTarget.DIRECT:
        return [DecodeTarget.DIRECT]

    if config.target == DecodeTarget.FILES:
        if not config.formats:
            raise ValueError(
                "EvaluationConfig.formats must be a non-empty sequence when target=FILES."
            )
        resolved: list[AudioFileFormat | DecodeTarget] = []
        for value in config.formats:
            fmt = AudioFileFormat.from_value(value)
            if fmt not in resolved:
                resolved.append(fmt)
        return resolved

    raise ValueError(f"Unsupported decode target: {config.target!r}")


def _output_path(
    input_path: Path,
    method: str,
    message_name: str,
    target: AudioFileFormat | DecodeTarget,
    config: EvaluationConfig,
) -> Path:
    if isinstance(target, DecodeTarget):
        return config.output_dir / "direct"

    filename = "{input_stem}_{input_hash}__{method}__{message}__{target}{suffix}".format(
        input_stem=_slug(input_path.stem),
        input_hash=_path_hash(input_path),
        method=_slug(method),
        message=_slug(message_name),
        target=target.value,
        suffix=target.extension,
    )
    return config.output_dir / filename


def _error_row(
    wav_file: WavFile,
    method: str,
    message: EvaluationMessage,
    target: AudioFileFormat | DecodeTarget,
    error: Exception,
    output_path: Path | None = None,
    failure_kind: str | None = None,
    attack: str | None = None,
    encode_time: float | None = None,
    attack_time: float | None = None,
    attack_parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    metric_errors: dict[str, str] | None = None,
) -> EvaluationRow:
    return EvaluationRow(
        input_path=wav_file.path,
        method=method,
        message_name=message.name,
        message_length=message.length,
        decode_mode=_decode_mode(target),
        format=_format_value(target),
        success=False,
        metrics=dict(metrics or {}),
        metric_errors=dict(metric_errors or {}),
        output_path=output_path if not isinstance(target, DecodeTarget) else None,
        error=str(error),
        failure_kind=failure_kind or FailureKind.ENCODE_ERROR,
        repetition=message.index,
        is_lossy=_is_lossy(target),
        transformation_name=_transformation_name(target),
        attack=attack,
        attack_parameters=dict(attack_parameters or {}),
        message_bits=list(message.bits),
        sample_rate=wav_file.samplerate,
        duration_seconds=len(wav_file.samples) / wav_file.samplerate if wav_file.samplerate else None,
        encode_time_seconds=encode_time,
        attack_time_seconds=attack_time,
    )


def _method_label(method: SteganographyMethod, method_spec: _MethodSpec) -> str:
    """The method's own name, plus its parameters when they differ from defaults.

    Two settings of one method are different treatments; without the
    parameters in the label their rows would be pooled into one group.
    """
    try:
        label = method.type()
    except Exception:
        return method_spec.label
    if method_spec.parameters:
        rendered = ", ".join(f"{key}={value}" for key, value in method_spec.parameters.items())
        label = f"{label} ({rendered})"
    return label


def _decode_mode(target: AudioFileFormat | DecodeTarget) -> str:
    if isinstance(target, DecodeTarget):
        return "direct"
    return "file_roundtrip"


def _format_value(target: AudioFileFormat | DecodeTarget) -> str | None:
    if isinstance(target, DecodeTarget):
        return None
    return target.value


def _is_lossy(target: AudioFileFormat | DecodeTarget) -> bool:
    if isinstance(target, DecodeTarget):
        return False
    return target.is_lossy


def _transformation_name(target: AudioFileFormat | DecodeTarget) -> str | None:
    if isinstance(target, DecodeTarget):
        return None
    if target.is_lossy:
        return "codec_compression"
    return "serialization_roundtrip"


def _validate_bits(bits: Sequence[int]) -> list[int]:
    values = [int(bit) for bit in bits]
    invalid = [bit for bit in values if bit not in {0, 1}]
    if invalid:
        raise ValueError("Evaluation messages must contain only 0 and 1 values.")
    return values


def _normalize_message(message: EvaluationMessage) -> EvaluationMessage:
    return EvaluationMessage(
        name=message.name,
        bits=tuple(_validate_bits(message.bits)),
        source=message.source,
        seed=message.seed,
        index=message.index,
        metadata=dict(message.metadata),
    )


def _generated_seeds(specs: Sequence[RandomMessageSpec], base_seed: int | None) -> list[int | None]:
    """One seed per message specification, keyed by its length.

    Keying by length rather than by position keeps the messages of one length
    fixed when other lengths are added to or removed from a sweep; two
    specifications of the same length are told apart by their order.
    """
    if base_seed is None:
        return [None for _ in specs]
    seeds: list[int | None] = []
    occurrences: dict[int, int] = {}
    for spec in specs:
        occurrence = occurrences.get(spec.length, 0)
        occurrences[spec.length] = occurrence + 1
        seeds.append(message_seed(base_seed, spec.length, occurrence))
    return seeds


def _validate_unique_message_names(messages: Sequence[EvaluationMessage]) -> None:
    names = [message.name for message in messages]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Evaluation message names must be unique: {sorted(duplicates)}")


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _callable_label(value: Callable[..., Any]) -> str:
    return getattr(value, "__name__", value.__class__.__name__)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"


def _path_hash(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]


__all__ = [
    # Re-exported dataclasses (now defined in sibling modules)
    "EvaluationConfig",
    "EvaluationMessage",
    "EvaluationResult",
    "EvaluationRow",
    "FailurePolicy",
    "RandomMessageSpec",
    # Engine entry points
    "available_attack_names",
    "evaluate_files",
    "evaluate_files_async",
    "load_files",
    "load_resource_files",
]
