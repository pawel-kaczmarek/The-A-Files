"""Core abstractions for robustness attacks.

An attack models a transformation a stego signal may undergo between embedding
and extraction: accidental processing, storage or transmission, format
conversion, acoustic playback and recapture, or a deliberate attempt to destroy
the payload while keeping the audio usable.

Every attack is a frozen dataclass whose fields *are* its parameters, so the
configuration and the operator are the same object and there are no hidden
constants. Applying one returns the processed audio together with metadata
describing exactly what was done, which is what makes a benchmark row
reproducible.

Conventions that hold for every attack in this package:

* Audio is processed in float64 with a nominal full scale of [-1, 1].
  Integer input is converted explicitly by its dtype peak and converted back
  afterwards; it is never reinterpreted as float amplitudes.
* Mono is a 1-D array; multi-channel is (samples, channels). Attacks are
  applied to each channel independently unless the attack documents otherwise,
  and nothing is silently downmixed.
* Stochastic attacks take a ``seed`` and are deterministic given one.
* Nothing is normalised behind the caller's back. Where an attack must limit
  amplitude it records that it did so in its metadata.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar

import numpy as np

from taf.models.card import AttackCard


class AttackCategory(str, Enum):
    """Scientific grouping of attacks by the phenomenon they model.

    Members are listed in presentation order: the catalogue groups attacks
    in this order.
    """

    NOISE = "noise"
    CODEC = "codec"
    FILTERING = "filtering"
    RESAMPLING = "resampling"
    QUANTIZATION = "quantization"
    AMPLITUDE = "amplitude"
    TEMPORAL = "temporal"
    ACOUSTIC = "acoustic"
    PIPELINE = "pipeline"


class Severity(str, Enum):
    """Named severity steps.

    A severity is only ever a label for a concrete parameter set: it is
    resolved through ``taf.attacks.presets`` into real numbers, which are the
    values recorded in results. Severity never travels into an attack in place
    of its parameters.
    """

    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


class AttackError(RuntimeError):
    """Raised when an attack cannot be carried out as configured."""


class AttackToolUnavailableError(AttackError):
    """Raised when an attack needs an external tool that is not installed."""


@dataclass(frozen=True)
class AttackResult:
    """Processed audio plus a full description of the transformation."""

    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def samples(self) -> np.ndarray:
        """Alias matching the WavFile vocabulary used elsewhere in the package."""
        return self.audio


def to_float(audio: np.ndarray) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Convert any supported input to float64 in [-1, 1].

    Returns the converted signal, a function restoring the original dtype, and
    metadata recording the conversion. Integer input is scaled by the dtype
    peak rather than reinterpreted, so an int16 cover and its float equivalent
    are attacked identically.
    """
    array = np.asarray(audio)
    if array.size == 0:
        raise AttackError("audio is empty")

    source_dtype = array.dtype
    info: dict[str, Any] = {"input_dtype": str(source_dtype)}

    if np.issubdtype(source_dtype, np.integer):
        iinfo = np.iinfo(source_dtype)
        peak = float(max(abs(iinfo.min), iinfo.max))
        info["integer_peak"] = peak
        converted = array.astype(np.float64) / peak

        def restore(processed: np.ndarray) -> np.ndarray:
            scaled = np.rint(np.clip(processed, -1.0, 1.0) * peak)
            return np.clip(scaled, iinfo.min, iinfo.max).astype(source_dtype)

    else:
        converted = array.astype(np.float64, copy=True)

        def restore(processed: np.ndarray) -> np.ndarray:
            return processed.astype(source_dtype, copy=False)

    if not np.all(np.isfinite(converted)):
        raise AttackError("audio contains NaN or Inf samples")

    return converted, restore, info


def as_columns(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """View any signal as (samples, channels); flag whether it was mono."""
    if audio.ndim == 1:
        return audio[:, None], True
    if audio.ndim == 2:
        return audio, False
    raise AttackError(f"audio must be 1-D or 2-D, got {audio.ndim} dimensions")


def from_columns(audio: np.ndarray, was_mono: bool) -> np.ndarray:
    """Undo :func:`as_columns`."""
    return audio[:, 0] if was_mono else audio


def per_channel(audio: np.ndarray, function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a mono operator to each channel, keeping the channel layout.

    Channels may come back longer or shorter (resampling, cropping); they are
    all the same length because the operator is deterministic per channel, and
    a mismatch is reported rather than silently padded.
    """
    columns, was_mono = as_columns(audio)
    processed = [np.asarray(function(columns[:, index])) for index in range(columns.shape[1])]

    lengths = {len(channel) for channel in processed}
    if len(lengths) != 1:
        raise AttackError(f"channels came back with different lengths: {sorted(lengths)}")

    return from_columns(np.stack(processed, axis=1), was_mono)


def signal_power(audio: np.ndarray) -> float:
    """Mean square of the signal, guarded against an all-zero input."""
    return float(np.mean(np.square(np.asarray(audio, dtype=np.float64))))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(signal_power(audio)))


def measured_snr_db(reference: np.ndarray, processed: np.ndarray) -> float:
    """SNR of ``processed`` against ``reference`` over their common length."""
    length = min(len(reference), len(processed))
    error = np.asarray(processed[:length], dtype=np.float64) - np.asarray(
        reference[:length], dtype=np.float64
    )
    noise_power = signal_power(error)
    if noise_power == 0.0:
        return float("inf")
    reference_power = signal_power(reference[:length])
    if reference_power == 0.0:
        return float("-inf")
    return 10.0 * np.log10(reference_power / noise_power)


def clip_to_full_scale(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip to [-1, 1] and report how many samples were affected."""
    clipped_count = int(np.count_nonzero(np.abs(audio) > 1.0))
    return np.clip(audio, -1.0, 1.0), clipped_count


def validate_cutoff(cutoff_hz: float, sample_rate: int, name: str = "cutoff_hz") -> None:
    """Reject cutoffs at or above Nyquist.

    A fixed 18 kHz low-pass is meaningless at a 16 kHz sampling rate, and
    scipy would either raise deep inside the filter design or silently produce
    nonsense; failing here says which parameter is wrong.
    """
    nyquist = sample_rate / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        raise AttackError(
            f"{name}={cutoff_hz} Hz must lie strictly between 0 and the Nyquist "
            f"frequency ({nyquist} Hz) for a {sample_rate} Hz signal"
        )


class Attack(ABC):
    """Base class for all attacks.

    Subclasses are frozen dataclasses declaring their parameters as fields and
    implementing :meth:`_process`. The base class handles dtype conversion,
    validation and metadata assembly, so every attack reports the same facts
    about what it did.
    """

    name: ClassVar[str] = "attack"
    category: ClassVar[AttackCategory] = AttackCategory.NOISE
    #: True when the attack may change the signal length or the sample rate,
    #: which desynchronises decoders that index by absolute position.
    changes_length_or_rate: ClassVar[bool] = False
    #: What the attack models, for the catalogue, the UI and the
    #: documentation (``taf.models.card``). Packaged attacks must declare one.
    card: ClassVar[AttackCard | None] = None

    @classmethod
    def severity_levels(cls, severity: Severity, sample_rate: int) -> dict[str, Any] | None:
        """Explicit parameters of this attack at ``severity``, or None.

        The packaged attacks have their levels in ``taf.attacks.presets``; an
        attack defined elsewhere - a plugin - overrides this to declare its
        own. Values that depend on the sampling rate (cutoffs, target rates)
        must be derived from ``sample_rate`` so they stay below Nyquist.
        """
        return None

    @abstractmethod
    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Transform float64 audio; return (audio, sample_rate, extra metadata)."""

    def parameters(self) -> dict[str, Any]:
        """The attack's configuration, as recorded in benchmark results."""
        values = asdict(self) if hasattr(self, "__dataclass_fields__") else {}
        return {key: _jsonable(value) for key, value in values.items()}

    def apply(self, audio: np.ndarray, sample_rate: int) -> AttackResult:
        """Run the attack and describe the result.

        The returned metadata always carries the attack name and category, its
        parameters, the input and output sample rates and lengths, the dtype
        conversion, and how much the signal changed in SNR terms when that is
        meaningful (same rate and length).
        """
        if sample_rate <= 0:
            raise AttackError(f"sample_rate must be positive, got {sample_rate}")

        converted, restore, conversion = to_float(audio)
        processed, output_rate, extra = self._process(converted, sample_rate)
        processed = np.asarray(processed, dtype=np.float64)

        if not np.all(np.isfinite(processed)):
            raise AttackError(f"attack '{self.name}' produced non-finite samples")

        metadata: dict[str, Any] = {
            "attack": self.name,
            "category": self.category.value,
            "parameters": self.parameters(),
            "input_sample_rate": sample_rate,
            "output_sample_rate": output_rate,
            "input_length": int(np.asarray(converted).shape[0]),
            "output_length": int(processed.shape[0]),
            "channels": 1 if processed.ndim == 1 else int(processed.shape[1]),
            **conversion,
            **extra,
        }

        if output_rate == sample_rate and metadata["input_length"] == metadata["output_length"]:
            metadata["measured_snr_db"] = measured_snr_db(converted, processed)

        return AttackResult(audio=restore(processed), sample_rate=output_rate, metadata=metadata)


def _jsonable(value: Any) -> Any:
    """Make parameter values safe for JSON/CSV result files."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def deprecated_alias(old: str, new: str) -> None:
    """Warn that an attack name has been renamed."""
    warnings.warn(
        f"attack '{old}' has been renamed to '{new}'; the old name will be removed",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "Attack",
    "AttackCategory",
    "AttackError",
    "AttackResult",
    "AttackToolUnavailableError",
    "Severity",
    "as_columns",
    "clip_to_full_scale",
    "deprecated_alias",
    "from_columns",
    "measured_snr_db",
    "per_channel",
    "rms",
    "signal_power",
    "to_float",
    "validate_cutoff",
]
