from __future__ import annotations

import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import numpy as np

from taf.audio.formats import AudioFileFormat
from taf.audio.io import load_audio, save_audio
from taf.models.WavFile import WavFile
from taf.models.types import MethodType, MetricType

MIN_MESSAGE_BITS = 4
MAX_MESSAGE_BITS = 120


@dataclass(frozen=True)
class MethodOption:
    method_type: MethodType
    name: str
    class_name: str
    description: str


@dataclass(frozen=True)
class MetricOption:
    metric_type: MetricType
    name: str
    class_name: str
    category: str
    description: str


@dataclass(frozen=True)
class AttackOption:
    name: str
    class_name: str


def bits_to_string(bits: Sequence[int]) -> str:
    return "".join(str(_validate_bit(bit)) for bit in bits)


def string_to_bits(value: str) -> list[int]:
    return [int(character) for character in value]


def validate_binary_message(value: str) -> list[int]:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Message cannot be empty.")
    if any(character not in {"0", "1"} for character in normalized):
        raise ValueError("Message can contain only 0 and 1.")
    if len(normalized) < MIN_MESSAGE_BITS or len(normalized) > MAX_MESSAGE_BITS:
        raise ValueError(
            f"Message length must be between {MIN_MESSAGE_BITS} and {MAX_MESSAGE_BITS} bits."
        )
    return string_to_bits(normalized)


def generate_message(length: int, seed: int | None = None) -> list[int]:
    if length < MIN_MESSAGE_BITS or length > MAX_MESSAGE_BITS:
        raise ValueError(
            f"Message length must be between {MIN_MESSAGE_BITS} and {MAX_MESSAGE_BITS} bits."
        )
    rng = np.random.default_rng(seed)
    return [int(bit) for bit in rng.integers(0, 2, size=length)]


def calculate_bit_accuracy(original: Sequence[int], decoded: Sequence[int]) -> float:
    if not original:
        return 0.0
    correct = calculate_correct_bits(original, decoded)
    return correct / len(original)


def calculate_correct_bits(original: Sequence[int], decoded: Sequence[int]) -> int:
    return sum(
        1
        for index, original_bit in enumerate(original)
        if index < len(decoded) and int(original_bit) == int(decoded[index])
    )


def load_uploaded_audio(uploaded_file: BinaryIO) -> tuple[np.ndarray, int]:
    file_name = getattr(uploaded_file, "name", "")
    suffix = Path(file_name).suffix.lower()
    if suffix not in {AudioFileFormat.WAV.extension, AudioFileFormat.FLAC.extension}:
        raise ValueError("Only WAV and FLAC files are supported.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(_uploaded_file_bytes(uploaded_file))

        wav_file = load_audio(temp_path)
        return wav_file.samples, wav_file.samplerate
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_audio_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = Path(temp_file.name)

        wav_file = WavFile(samplerate=sample_rate, samples=samples, path=temp_path)
        save_audio(wav_file, temp_path, AudioFileFormat.WAV)
        return temp_path.read_bytes()
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def discover_methods(sample_rate: int = 16000) -> list[dict[str, str]]:
    methods = _method_map(sample_rate)
    return [
        {
            "name": method_type.name,
            "type": _safe_method_description(method),
            "class": method.__class__.__name__,
        }
        for method_type, method in sorted(methods.items(), key=lambda item: item[0].name)
    ]


def quick_method_options(sample_rate: int) -> list[MethodOption]:
    methods = _method_map(sample_rate)
    options: list[MethodOption] = []
    for method_type, method in sorted(methods.items(), key=lambda item: item[0].name):
        options.append(
            MethodOption(
                method_type=method_type,
                name=method_type.name,
                class_name=method.__class__.__name__,
                description=_safe_method_description(method),
            )
        )
    return options


def discover_metrics() -> list[dict[str, str]]:
    from taf.metrics.factory import MetricFactory

    metrics = MetricFactory._all_methods()
    return [
        {
            "name": metric_type.name,
            "class": metric.__class__.__name__,
            "category": _metric_category(metric.__class__.__module__),
        }
        for metric_type, metric in sorted(metrics.items(), key=lambda item: item[0].name)
    ]


def quick_metric_options() -> list[MetricOption]:
    metrics = _metric_map()
    return [
        MetricOption(
            metric_type=metric_type,
            name=metric_type.name,
            class_name=metric.__class__.__name__,
            category=_metric_category(metric.__class__.__module__),
            description=_safe_metric_name(metric),
        )
        for metric_type, metric in sorted(metrics.items(), key=lambda item: item[0].name)
    ]


def quick_attack_options() -> list[AttackOption]:
    from taf.attacks.attacks import CorruptedWavFile

    return [
        AttackOption(name=row["name"], class_name=CorruptedWavFile.__name__)
        for row in discover_attacks()
    ]


def discover_attacks() -> list[dict[str, str]]:
    from taf.attacks.attacks import CorruptedWavFile

    attack_rows: list[dict[str, str]] = []
    for name, member in inspect.getmembers(CorruptedWavFile, predicate=inspect.isfunction):
        if name.startswith("_") or name == "__init__":
            continue
        if name not in CorruptedWavFile.__dict__:
            continue
        attack_rows.append(
            {
                "name": name,
                "class": CorruptedWavFile.__name__,
            }
        )
    return sorted(attack_rows, key=lambda row: row["name"])


def create_method(sample_rate: int, method_type: MethodType):
    from taf.methods.factory import SteganographyMethodFactory

    return SteganographyMethodFactory.get(sample_rate, method_type)


def create_metric(metric_type: MetricType):
    from taf.metrics.factory import MetricFactory

    return MetricFactory.get(metric_type)


def apply_attacks(
    samples: np.ndarray,
    sample_rate: int,
    attack_options: Sequence[AttackOption],
) -> tuple[np.ndarray, int]:
    from taf.attacks.attacks import CorruptedWavFile

    corrupted = CorruptedWavFile(
        WavFile(samplerate=sample_rate, samples=np.asarray(samples).copy(), path=Path(""))
    )
    for option in attack_options:
        attack = getattr(corrupted, option.name)
        corrupted = attack()
    return np.asarray(corrupted.samples), corrupted.samplerate


def to_mono(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    if samples.ndim <= 1:
        return samples, False
    return np.mean(samples, axis=1).astype(samples.dtype, copy=False), True


def _method_map(sample_rate: int):
    from taf.methods.factory import SteganographyMethodFactory

    return SteganographyMethodFactory._all_methods(sample_rate)


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


def _metric_map():
    from taf.metrics.factory import MetricFactory

    return MetricFactory._all_methods()


def _safe_method_description(method: object) -> str:
    type_function = getattr(method, "type", None)
    if not callable(type_function):
        return ""
    try:
        return str(type_function())
    except Exception:
        return ""


def _safe_metric_name(metric: object) -> str:
    name_function = getattr(metric, "name", None)
    if not callable(name_function):
        return ""
    try:
        return str(name_function())
    except Exception:
        return ""


def _uploaded_file_bytes(uploaded_file: BinaryIO) -> bytes:
    getbuffer = getattr(uploaded_file, "getbuffer", None)
    if callable(getbuffer):
        return bytes(getbuffer())

    position = None
    tell = getattr(uploaded_file, "tell", None)
    seek = getattr(uploaded_file, "seek", None)
    if callable(tell):
        position = tell()
    if callable(seek):
        seek(0)

    content = uploaded_file.read()
    if callable(seek) and position is not None:
        seek(position)
    return content


def _validate_bit(bit: int) -> int:
    normalized = int(bit)
    if normalized not in {0, 1}:
        raise ValueError("Bits must be 0 or 1.")
    return normalized
