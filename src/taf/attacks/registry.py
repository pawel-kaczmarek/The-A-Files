"""Attack lookup and construction from names or specification strings.

The registry is what lets an experiment configuration stay textual - a YAML
file, a CLI flag, an HTTP request body - while still producing a fully
parameterised attack. A specification is either

* a bare name:            ``"awgn"``                  (attack defaults)
* a name with parameters: ``"awgn:snr_db=20,seed=7"``
* a name with severity:   ``"awgn@strong"``           (resolved via presets)
* a benchmark pipeline:   ``"pipeline:streaming_upload"``

Parameters are parsed with Python literal rules, so ``seed=None``,
``zero_phase=False`` and ``bitrate_kbps=64`` all arrive with the right type
rather than as strings.
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Iterable

from taf.attacks.acoustic import AcousticChannel, EchoAttack, Reverberation
from taf.attacks.amplitude import Clipping, DynamicRangeCompression, GainChange
from taf.attacks.base import Attack, AttackError, Severity
from taf.attacks.codec import CodecCompression
from taf.attacks.filtering import (
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
    MovingAverageSmoothing,
    NotchFilter,
)
from taf.attacks.noise import AdditivePinkNoise, AdditiveWhiteNoise, ImpulseNoise
from taf.attacks.quantization import BitDepthReduction
from taf.attacks.resampling import ResampleRoundTrip, SampleRateOffset
from taf.attacks.temporal import (
    Cropping,
    PitchShift,
    SampleDropout,
    SampleInsertionDeletion,
    SpeedChange,
    TimeShift,
    TimeStretch,
    ZeroPadding,
)

ATTACK_CLASSES: dict[str, type[Attack]] = {
    cls.name: cls
    for cls in (
        AdditiveWhiteNoise,
        AdditivePinkNoise,
        ImpulseNoise,
        CodecCompression,
        LowPassFilter,
        HighPassFilter,
        BandPassFilter,
        NotchFilter,
        MovingAverageSmoothing,
        ResampleRoundTrip,
        SampleRateOffset,
        BitDepthReduction,
        GainChange,
        Clipping,
        DynamicRangeCompression,
        TimeShift,
        Cropping,
        SampleInsertionDeletion,
        SampleDropout,
        TimeStretch,
        SpeedChange,
        PitchShift,
        ZeroPadding,
        EchoAttack,
        Reverberation,
        AcousticChannel,
    )
}

#: Convenience names that fix one field of a general attack, so that common
#: cases read naturally in a configuration file.
ATTACK_FACTORIES: dict[str, Callable[..., Attack]] = {
    "mp3": lambda **kwargs: CodecCompression(codec="mp3", **kwargs),
    "aac": lambda **kwargs: CodecCompression(codec="aac", **kwargs),
    "opus": lambda **kwargs: CodecCompression(codec="opus", **kwargs),
    "vorbis": lambda **kwargs: CodecCompression(codec="vorbis", **kwargs),
}


#: Names used before the attacks were reworked, kept so that configurations
#: and saved experiment definitions written against the old API keep running.
#: They resolve to the corrected implementation, which in several cases behaves
#: differently from the original: ``additive_noise`` is now specified by SNR,
#: ``resample`` is a round trip, and ``frequency_filter`` removes a real band.
LEGACY_ALIASES: dict[str, str] = {
    "additive_noise": "awgn",
    "amplitude_scaling": "gain",
    "low_pass_filter": "low_pass",
    "high_pass_filter": "high_pass",
    "band_pass_filter": "band_pass",
    "frequency_filter": "notch",
    "notch_filter": "notch",
    "flip_random_samples": "impulse_noise",
    "cut_random_samples": "dropout",
    "sample_suppression": "dropout",
    "quantization": "bit_depth",
    "mp3_compression": "mp3",
    "aac_compression": "aac",
    "opus_compression": "opus",
    "vorbis_compression": "vorbis",
    "echo_addition": "echo",
    "reverberation": "reverb",
    "speed_change": "speed",
    "dynamic_range_compression": "compression_dynamic",
}


def resolve_name(name: str) -> str:
    """Canonical attack name, following legacy aliases."""
    return LEGACY_ALIASES.get(name, name)


def available_attacks() -> list[str]:
    """Every attack name the registry can build, sorted."""
    return sorted(set(ATTACK_CLASSES) | set(ATTACK_FACTORIES))


def attack_class(name: str) -> type[Attack]:
    """The class implementing ``name``, following codec and legacy aliases."""
    name = resolve_name(name)
    if name in ATTACK_CLASSES:
        return ATTACK_CLASSES[name]
    if name in ATTACK_FACTORIES:
        return CodecCompression
    raise AttackError(f"unknown attack {name!r}; known: {available_attacks()}")


def create(name: str, **parameters: Any) -> Attack:
    """Build an attack by name (canonical or legacy) with explicit parameters."""
    name = resolve_name(name)
    if name in ATTACK_FACTORIES:
        factory = ATTACK_FACTORIES[name]
    elif name in ATTACK_CLASSES:
        factory = ATTACK_CLASSES[name]
    else:
        raise AttackError(f"unknown attack {name!r}; known: {available_attacks()}")

    try:
        return factory(**parameters)
    except TypeError as error:
        raise AttackError(f"invalid parameters for attack {name!r}: {error}") from error


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # Bare words such as codec=mp3 or mode=peak are strings.
        return text


def parse_spec(spec: str) -> tuple[str, dict[str, Any], Severity | None]:
    """Split a specification string into (name, parameters, severity)."""
    if not isinstance(spec, str) or not spec.strip():
        raise AttackError(f"attack specification must be a non-empty string, got {spec!r}")

    text = spec.strip()
    parameters: dict[str, Any] = {}
    severity: Severity | None = None

    if ":" in text:
        text, _, parameter_text = text.partition(":")
        for chunk in parameter_text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise AttackError(
                    f"parameter {chunk!r} in {spec!r} must be written as key=value"
                )
            key, _, value = chunk.partition("=")
            parameters[key.strip()] = _parse_value(value.strip())

    if "@" in text:
        text, _, severity_text = text.partition("@")
        try:
            severity = Severity(severity_text.strip().lower())
        except ValueError as error:
            raise AttackError(
                f"unknown severity {severity_text!r} in {spec!r}; "
                f"expected one of {[level.value for level in Severity]}"
            ) from error

    return text.strip(), parameters, severity


def unknown_specs(specs: Iterable[str]) -> list[str]:
    """Specifications that cannot be built, for validation messages.

    Shared by the evaluation workflow and the experiment schema so that a
    configuration accepted by one is accepted by the other. It understands the
    full specification grammar - parameters, severities, pipelines and legacy
    aliases - rather than comparing against a list of bare names.
    """
    known = set(available_attacks()) | set(LEGACY_ALIASES) | {"pipeline"}
    unknown: list[str] = []
    for spec in specs:
        if isinstance(spec, Attack):
            continue
        try:
            name, _, _ = parse_spec(str(spec))
        except AttackError:
            unknown.append(str(spec))
            continue
        if name not in known:
            unknown.append(str(spec))
    return unknown


def build(spec: str | Attack, sample_rate: int | None = None) -> Attack:
    """Build an attack from a specification string.

    ``sample_rate`` is needed only for severity presets, whose parameters may
    depend on it - a low-pass cutoff has to stay below Nyquist, and a
    resampling target below the source rate.
    """
    if isinstance(spec, Attack):
        return spec

    name, parameters, severity = parse_spec(spec)

    if name == "pipeline":
        from taf.attacks.presets import benchmark_pipeline

        pipeline_name = parameters.pop("name", None)
        if pipeline_name is None:
            raise AttackError(
                "a pipeline specification needs a name, e.g. 'pipeline:name=streaming_upload'"
            )
        return benchmark_pipeline(str(pipeline_name), sample_rate=sample_rate)

    name = resolve_name(name)

    if severity is not None:
        from taf.attacks.presets import severity_parameters

        preset = severity_parameters(name, severity, sample_rate=sample_rate)
        # Explicit parameters win over the preset, so a sweep can pin one
        # field while keeping the rest of a severity level.
        preset.update(parameters)
        parameters = preset

    return create(name, **parameters)


def build_all(specs: Iterable[str | Attack], sample_rate: int | None = None) -> list[Attack]:
    return [build(spec, sample_rate) for spec in specs]


__all__ = [
    "ATTACK_CLASSES",
    "ATTACK_FACTORIES",
    "LEGACY_ALIASES",
    "resolve_name",
    "attack_class",
    "available_attacks",
    "build",
    "build_all",
    "create",
    "parse_spec",
    "unknown_specs",
]
