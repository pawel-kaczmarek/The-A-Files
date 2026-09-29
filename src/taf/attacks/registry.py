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
from taf.models.card import AttackCard, fallback_card
from taf.attacks.codec import CodecCompression, codec_shortcut_card
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
    # libvorbis rejects 128 kbit/s below 32 kHz; 64 kbit/s encodes at 16-48 kHz.
    "vorbis": lambda **kwargs: CodecCompression(codec="vorbis", **{"bitrate_kbps": 64, **kwargs}),
}

#: Cards of the convenience names, which have no class of their own.
ATTACK_FACTORY_CARDS: dict[str, AttackCard] = {name: codec_shortcut_card(name) for name in ATTACK_FACTORIES}


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


def attack_classes() -> dict[str, type[Attack]]:
    """Packaged attack classes plus those registered by other distributions.

    Plugins come from the ``taf.attacks`` entry-point group (see
    ``taf.plugins``). An entry that is not an ``Attack`` subclass, or that
    reuses a packaged name, is skipped with a warning.
    """
    from loguru import logger

    from taf.plugins import ATTACK_GROUP, load_entry_points

    merged = dict(ATTACK_CLASSES)
    for name, cls in load_entry_points(ATTACK_GROUP).items():
        if name in merged or name in ATTACK_FACTORIES or name in LEGACY_ALIASES:
            logger.warning("Attack plugin {!r} shadows a packaged name; ignored", name)
            continue
        if not (isinstance(cls, type) and issubclass(cls, Attack)):
            logger.warning("Attack plugin {!r} is not an Attack subclass; ignored", name)
            continue
        merged[name] = cls
    return merged


def available_attacks() -> list[str]:
    """Every attack name the registry can build, sorted."""
    return sorted(set(attack_classes()) | set(ATTACK_FACTORIES))


def attack_class(name: str) -> type[Attack]:
    """The class implementing ``name``, following codec and legacy aliases."""
    name = resolve_name(name)
    classes = attack_classes()
    if name in classes:
        return classes[name]
    if name in ATTACK_FACTORIES:
        return CodecCompression
    raise AttackError(f"unknown attack {name!r}; known: {available_attacks()}")


def attack_card(name: str) -> AttackCard:
    """What the attack ``name`` models: its card, or one made from its docstring."""
    name = resolve_name(name)
    if name in ATTACK_FACTORY_CARDS:
        return ATTACK_FACTORY_CARDS[name]
    cls = attack_class(name)
    card = getattr(cls, "card", None)
    return card if isinstance(card, AttackCard) else fallback_card(cls, name, AttackCard)


def create(name: str, **parameters: Any) -> Attack:
    """Build an attack by name (canonical or legacy) with explicit parameters."""
    name = resolve_name(name)
    classes = attack_classes()
    if name in ATTACK_FACTORIES:
        factory = ATTACK_FACTORIES[name]
    elif name in classes:
        factory = classes[name]
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


def has_explicit_seed(spec: str | Attack) -> bool:
    """Whether a specification pins its own seed (``"awgn:seed=7"``)."""
    if isinstance(spec, Attack):
        return True
    _, parameters, _ = parse_spec(spec)
    return "seed" in parameters


def reseed(attack: Attack, seed: int) -> Attack:
    """A copy of ``attack`` whose random draws follow ``seed``.

    An attack built from a specification keeps the seed of its dataclass
    default, so every trial of an experiment would otherwise see the same
    noise realisation. The evaluation derives one seed per trial and applies
    it here. Deterministic attacks have no ``seed`` field and are returned
    unchanged; each stage of a pipeline gets its own child seed, so two noise
    stages in one pipeline do not draw the same sequence.
    """
    from dataclasses import fields, replace

    import numpy as np

    from taf.attacks.pipeline import AttackPipeline

    if isinstance(attack, AttackPipeline):
        children = np.random.SeedSequence(seed).spawn(len(attack.stages))
        stages = tuple(
            reseed(stage, int(child.generate_state(1)[0]))
            for stage, child in zip(attack.stages, children)
        )
        return replace(attack, stages=stages)

    if any(field.name == "seed" for field in fields(attack)):
        return replace(attack, seed=seed)
    return attack


__all__ = [
    "ATTACK_CLASSES",
    "ATTACK_FACTORIES",
    "ATTACK_FACTORY_CARDS",
    "LEGACY_ALIASES",
    "resolve_name",
    "attack_card",
    "attack_class",
    "attack_classes",
    "available_attacks",
    "build",
    "build_all",
    "create",
    "has_explicit_seed",
    "parse_spec",
    "reseed",
    "unknown_specs",
]
