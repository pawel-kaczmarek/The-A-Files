from __future__ import annotations

import asyncio
import copy
import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from audio.formats import AudioFileFormat, DecodeTarget, is_direct_target
from audio.io import load_audio, save_audio
from loaders.WavFileLoader import WavFileLoader
from models.Metric import Metric
from models.SteganographyMethod import SteganographyMethod
from models.WavFile import WavFile
from models.types import MethodType, MetricType


class FailurePolicy(str, Enum):
    RECORD = "record"
    RAISE = "raise"


@dataclass(frozen=True)
class EvaluationMessage:
    name: str
    bits: tuple[int, ...]
    source: str = "manual"
    seed: int | None = None
    index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.bits)


@dataclass(frozen=True)
class RandomMessageSpec:
    length: int
    count: int = 1
    seed: int | None = None
    name_prefix: str = "random"


@dataclass
class EvaluationConfig:
    methods: Sequence[MethodType | SteganographyMethod | Callable[[int], SteganographyMethod]] | None = None
    metrics: Sequence[MetricType | Metric | Callable[[], Metric]] | None = None
    formats: Sequence[AudioFileFormat | DecodeTarget | str] = (DecodeTarget.DIRECT,)
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


@dataclass
class EvaluationRow:
    input_path: Path
    method: str
    message_name: str
    message_length: int
    decode_mode: str
    format: str | None
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    metric_errors: dict[str, str] = field(default_factory=dict)
    output_path: Path | None = None
    decoded_message: list[int] | None = None
    error: str | None = None
    is_lossy: bool = False
    transformation_name: str | None = None
    codec_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "method": self.method,
            "message_name": self.message_name,
            "message_length": self.message_length,
            "decode_mode": self.decode_mode,
            "format": self.format,
            "success": self.success,
            "metrics": self.metrics,
            "metric_errors": self.metric_errors,
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "decoded_message": self.decoded_message,
            "error": self.error,
            "is_lossy": self.is_lossy,
            "transformation_name": self.transformation_name,
            "codec_options": self.codec_options,
        }


@dataclass
class EvaluationResult:
    messages: dict[str, EvaluationMessage]
    rows: list[EvaluationRow] = field(default_factory=list)

    def success_rate(self) -> float:
        if not self.rows:
            return 0.0
        return sum(row.success for row in self.rows) / len(self.rows)

    def by_method(self) -> dict[str, list[EvaluationRow]]:
        grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
        for row in self.rows:
            grouped[row.method].append(row)
        return dict(grouped)

    def by_format(self) -> dict[str, list[EvaluationRow]]:
        grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
        for row in self.rows:
            grouped[row.format or DecodeTarget.DIRECT.value].append(row)
        return dict(grouped)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.rows]


@dataclass(frozen=True)
class _MethodSpec:
    label: str
    create: Callable[[int], SteganographyMethod]


@dataclass(frozen=True)
class _MetricSpec:
    label: str
    create: Callable[[], Metric]


def load_files(path: str | Path) -> list[WavFile]:
    directory = Path(path)
    loader = WavFileLoader(directory)
    files = sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in {AudioFileFormat.FLAC.extension, AudioFileFormat.WAV.extension}
    )
    return [loader.load(file_path.name) for file_path in files]


def load_resource_files(paths: Iterable[Path]) -> list[WavFile]:
    return [load_audio(path) for path in paths]


def apply_all_attacks(wav_file: WavFile):
    from attacks.attacks import CorruptedWavFile

    return (
        CorruptedWavFile(wav_file)
        .low_pass_filter()
        .resample()
        .pitch_shift()
        .additive_noise()
        .time_stretch()
        .frequency_filter()
        .flip_random_samples()
        .cut_random_samples()
        .amplitude_scaling()
    )


def evaluate_files(files: Iterable[WavFile], config: EvaluationConfig | None = None) -> EvaluationResult:
    return asyncio.run(evaluate_files_async(files, config))


async def evaluate_files_async(
    files: Iterable[WavFile],
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    resolved_config = config or EvaluationConfig()
    messages = _materialize_messages(resolved_config)
    method_specs = _method_specs(resolved_config)
    metric_specs = _metric_specs(resolved_config)
    targets = _normalize_targets(resolved_config.formats)
    semaphore = asyncio.Semaphore(max(1, resolved_config.max_workers))

    async def guarded_job(wav_file: WavFile, method_spec: _MethodSpec, message: EvaluationMessage):
        async with semaphore:
            return await asyncio.to_thread(
                _evaluate_file_method_message,
                wav_file,
                method_spec,
                message,
                metric_specs,
                targets,
                resolved_config,
            )

    tasks = [
        guarded_job(wav_file, method_spec, message)
        for wav_file in files
        for method_spec in method_specs
        for message in messages
    ]
    row_groups = await asyncio.gather(*tasks)
    rows = [row for group in row_groups for row in group]
    return EvaluationResult(messages={message.name: message for message in messages}, rows=rows)


def _evaluate_file_method_message(
    wav_file: WavFile,
    method_spec: _MethodSpec,
    message: EvaluationMessage,
    metric_specs: Sequence[_MetricSpec],
    targets: Sequence[AudioFileFormat | DecodeTarget],
    config: EvaluationConfig,
) -> list[EvaluationRow]:
    method = method_spec.create(wav_file.samplerate)
    method_label = _method_label(method, method_spec)

    try:
        encoded_samples = method.encode(wav_file.samples.copy(), list(message.bits))
    except Exception as error:
        if config.failure_policy == FailurePolicy.RAISE:
            raise
        return [
            _error_row(
                wav_file=wav_file,
                method=method_label,
                message=message,
                target=target,
                error=error,
            )
            for target in targets
        ]

    encoded = WavFile(samplerate=wav_file.samplerate, samples=encoded_samples, path=wav_file.path)
    return [
        _evaluate_target(wav_file, encoded, method, method_label, message, metric_specs, target, config)
        for target in targets
    ]


def _evaluate_target(
    original: WavFile,
    encoded: WavFile,
    method: SteganographyMethod,
    method_label: str,
    message: EvaluationMessage,
    metric_specs: Sequence[_MetricSpec],
    target: AudioFileFormat | DecodeTarget,
    config: EvaluationConfig,
) -> EvaluationRow:
    output_path = _output_path(original.path, method_label, message.name, target, config)

    try:
        if isinstance(target, DecodeTarget):
            decode_input = encoded
            output_path = None
        else:
            if output_path.exists() and not config.overwrite:
                raise FileExistsError(f"Output file already exists: {output_path}")
            options = config.codec_options.get(target.value, {})
            saved_path = save_audio(encoded, output_path, target, options)
            decode_input = load_audio(saved_path)
            if not config.keep_files:
                _remove_file(saved_path)
                output_path = None

        decoded_message = method.decode(decode_input.samples, message.length)
        success = bool(np.array_equal(list(message.bits), decoded_message))
        metrics, metric_errors = _calculate_metrics(original.samples, decode_input.samples, original.samplerate, metric_specs)
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
            output_path=output_path,
            decoded_message=list(decoded_message),
            is_lossy=_is_lossy(target),
            transformation_name=_transformation_name(target),
            codec_options=config.codec_options.get(_format_value(target) or "", {}),
        )
    except Exception as error:
        if config.failure_policy == FailurePolicy.RAISE:
            raise
        return _error_row(original, method_label, message, target, error, output_path)


def _calculate_metrics(
    original_samples: np.ndarray,
    processed_samples: np.ndarray,
    samplerate: int,
    metric_specs: Sequence[_MetricSpec],
) -> tuple[dict[str, Any], dict[str, str]]:
    metrics: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for metric_spec in metric_specs:
        metric = metric_spec.create()
        name = metric.name()
        try:
            metrics[name] = metric.calculate(original_samples, processed_samples, samplerate, 0.03, 0.75)
        except Exception as error:
            errors[name] = str(error)
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
        from metrics.factory import MetricFactory

        return [_MetricSpec(label=metric.name(), create=lambda original=metric: copy.deepcopy(original)) for metric in MetricFactory.get_all()]

    specs: list[_MetricSpec] = []
    for metric in config.metrics:
        if isinstance(metric, MetricType):
            specs.append(_metric_type_spec(metric))
        elif isinstance(metric, Metric):
            specs.append(_MetricSpec(label=metric.name(), create=lambda original=metric: copy.deepcopy(original)))
        elif callable(metric):
            specs.append(_MetricSpec(label=_callable_label(metric), create=metric))
        else:
            raise TypeError(f"Unsupported metric config entry: {metric!r}")
    return specs


def _method_type_spec(method_type: MethodType) -> _MethodSpec:
    def create(samplerate: int) -> SteganographyMethod:
        from methods.factory import SteganographyMethodFactory

        method = SteganographyMethodFactory.get(samplerate, method_type)
        if method is None:
            raise ValueError(f"Unknown steganography method: {method_type}")
        return method

    return _MethodSpec(label=method_type.name, create=create)


def _metric_type_spec(metric_type: MetricType) -> _MetricSpec:
    def create() -> Metric:
        from metrics.factory import MetricFactory

        metric = MetricFactory.get(metric_type)
        if metric is None:
            raise ValueError(f"Unknown metric: {metric_type}")
        return metric

    return _MetricSpec(label=metric_type.name, create=create)


def _normalize_targets(
    formats: Sequence[AudioFileFormat | DecodeTarget | str],
) -> list[AudioFileFormat | DecodeTarget]:
    if not formats:
        return [DecodeTarget.DIRECT]

    targets: list[AudioFileFormat | DecodeTarget] = []
    for value in formats:
        if is_direct_target(value):
            target: AudioFileFormat | DecodeTarget = DecodeTarget.DIRECT
        else:
            target = AudioFileFormat.from_value(value)
        if target not in targets:
            targets.append(target)
    return targets


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
) -> EvaluationRow:
    return EvaluationRow(
        input_path=wav_file.path,
        method=method,
        message_name=message.name,
        message_length=message.length,
        decode_mode=_decode_mode(target),
        format=_format_value(target),
        success=False,
        output_path=output_path if not isinstance(target, DecodeTarget) else None,
        error=str(error),
        is_lossy=_is_lossy(target),
        transformation_name=_transformation_name(target),
    )


def _method_label(method: SteganographyMethod, method_spec: _MethodSpec) -> str:
    try:
        return method.type()
    except Exception:
        return method_spec.label


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
    if not specs:
        return []
    if base_seed is None:
        return [None for _ in specs]
    seed_sequence = np.random.SeedSequence(base_seed)
    return [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(len(specs))]


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
