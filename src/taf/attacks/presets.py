"""Severity levels, benchmark suites and named channel pipelines.

Two rules govern everything in this module.

**A severity is never a parameter.** ``MILD`` and ``EXTREME`` are labels for
concrete numbers that are resolved here and written into the result row. A
result that recorded only "strong" would not be reproducible by anyone who
did not have this file at the same revision.

**Parameters are derived from the signal where the signal constrains them.**
A filter cutoff or a resampling target that is fixed in Hertz is wrong at some
sampling rate: an 18 kHz low-pass does nothing to 16 kHz speech, and a
"downsample to 22.05 kHz" attack is an *up*-sampling for a 16 kHz file and
therefore no attack at all. Those families are expressed relative to the
Nyquist frequency or chosen from a ladder filtered against the source rate.

Parameter choices are engineering benchmark choices informed by common
practice in the audio watermarking literature, stated here with the reason
they were picked. Where a number is conventional rather than derived, it is
described as such rather than attributed to a publication.
"""
from __future__ import annotations

from typing import Any, Sequence

from taf.attacks.base import Attack, AttackError, Severity
from taf.attacks.pipeline import AttackPipeline, chain
from taf.attacks.registry import build, create

DEFAULT_SAMPLE_RATE = 16000

#: Ladder of standard sampling rates used to pick resampling targets.
STANDARD_RATES: tuple[int, ...] = (48000, 44100, 32000, 22050, 16000, 8000)

#: SNR ladder shared by the three additive-noise processes. 40 dB is at the
#: edge of audibility on speech, 25 dB is clearly audible but undamaging, 15 dB
#: is a poor channel, and 5 dB is noise comparable to the signal itself - past
#: the point where the audio is worth protecting, included to show where a
#: method finally fails.
_NOISE_SNR_DB = {
    Severity.MILD: 40.0,
    Severity.MODERATE: 25.0,
    Severity.STRONG: 15.0,
    Severity.EXTREME: 5.0,
}

#: Cutoffs as a fraction of the Nyquist frequency, so they are valid at any
#: sampling rate. At 16 kHz these are 7.2, 4.0, 2.4 and 1.2 kHz.
_LOW_PASS_NYQUIST_FRACTION = {
    Severity.MILD: 0.90,
    Severity.MODERATE: 0.50,
    Severity.STRONG: 0.30,
    Severity.EXTREME: 0.15,
}

#: High-pass corners in Hz. These are absolute because they describe the low
#: end, which does not move with the sampling rate: 20 Hz is a DC blocker,
#: 300 Hz is the telephone band edge, 1 kHz removes the whole of the speech
#: fundamental range.
_HIGH_PASS_HZ = {
    Severity.MILD: 20.0,
    Severity.MODERATE: 100.0,
    Severity.STRONG: 300.0,
    Severity.EXTREME: 1000.0,
}

_BIT_DEPTH = {
    Severity.MILD: 16,
    Severity.MODERATE: 12,
    Severity.STRONG: 10,
    Severity.EXTREME: 8,
}

#: Gain in dB. Attenuation dominates the ladder because it is the common case;
#: the extreme step is a boost, which additionally drives peaks into clipping
#: and so is genuinely the most destructive of the four.
_GAIN_DB = {
    Severity.MILD: -3.0,
    Severity.MODERATE: -6.0,
    Severity.STRONG: -12.0,
    Severity.EXTREME: 6.0,
}

_CODEC_BITRATE_KBPS = {
    "mp3": {Severity.MILD: 256, Severity.MODERATE: 128, Severity.STRONG: 96, Severity.EXTREME: 64},
    "aac": {Severity.MILD: 192, Severity.MODERATE: 128, Severity.STRONG: 96, Severity.EXTREME: 64},
    "opus": {Severity.MILD: 128, Severity.MODERATE: 96, Severity.STRONG: 64, Severity.EXTREME: 32},
    "vorbis": {Severity.MILD: 192, Severity.MODERATE: 128, Severity.STRONG: 96, Severity.EXTREME: 64},
}

_CROP_FRACTION = {
    Severity.MILD: 0.001,
    Severity.MODERATE: 0.01,
    Severity.STRONG: 0.05,
    Severity.EXTREME: 0.10,
}

#: Time shifts in milliseconds. One millisecond is 16 samples at 16 kHz -
#: inaudible, and already fatal to a decoder that indexes by sample.
_SHIFT_MS = {
    Severity.MILD: 1.0,
    Severity.MODERATE: 10.0,
    Severity.STRONG: 100.0,
    Severity.EXTREME: 250.0,
}

#: Tempo and speed factors. Deviations of 1-2% are the ones that matter:
#: they are inaudible to most listeners yet accumulate into a large sample
#: offset across a clip. Larger factors are audible and so less realistic as
#: covert attacks.
_RATE_FACTOR = {
    Severity.MILD: 1.01,
    Severity.MODERATE: 0.99,
    Severity.STRONG: 1.05,
    Severity.EXTREME: 0.95,
}

_PITCH_SEMITONES = {
    Severity.MILD: 0.25,
    Severity.MODERATE: 0.5,
    Severity.STRONG: 1.0,
    Severity.EXTREME: 2.0,
}

#: (delay_ms, attenuation). Short quiet echoes colour the sound; long loud
#: ones are audible repetitions.
_ECHO = {
    Severity.MILD: (5.0, 0.1),
    Severity.MODERATE: (25.0, 0.25),
    Severity.STRONG: (50.0, 0.5),
    Severity.EXTREME: (100.0, 0.5),
}

#: RT60 in seconds: a treated room, a living room, a hall, a large hall.
_RT60 = {
    Severity.MILD: 0.2,
    Severity.MODERATE: 0.5,
    Severity.STRONG: 1.0,
    Severity.EXTREME: 1.5,
}

_CLIPPING_THRESHOLD = {
    Severity.MILD: 0.95,
    Severity.MODERATE: 0.90,
    Severity.STRONG: 0.80,
    Severity.EXTREME: 0.70,
}

_DROPOUT_FRACTION = {
    Severity.MILD: 0.001,
    Severity.MODERATE: 0.005,
    Severity.STRONG: 0.01,
    Severity.EXTREME: 0.05,
}

_JITTER_EVENTS = {
    Severity.MILD: 2,
    Severity.MODERATE: 10,
    Severity.STRONG: 50,
    Severity.EXTREME: 200,
}

#: Clock offsets in parts per million. Consumer sound cards routinely differ
#: by tens to hundreds of ppm; 500 ppm is a bad pairing.
_CLOCK_PPM = {
    Severity.MILD: 10.0,
    Severity.MODERATE: 50.0,
    Severity.STRONG: 100.0,
    Severity.EXTREME: 500.0,
}

_SMOOTHING_WINDOW = {
    Severity.MILD: 3,
    Severity.MODERATE: 5,
    Severity.STRONG: 9,
    Severity.EXTREME: 17,
}

_COMPRESSION_RATIO = {
    Severity.MILD: 2.0,
    Severity.MODERATE: 4.0,
    Severity.STRONG: 8.0,
    Severity.EXTREME: 16.0,
}

#: Notch quality factor: a high Q is a narrow notch, so severity widens it.
_NOTCH_QUALITY = {
    Severity.MILD: 60.0,
    Severity.MODERATE: 30.0,
    Severity.STRONG: 10.0,
    Severity.EXTREME: 3.0,
}


def resampling_targets(sample_rate: int) -> list[int]:
    """Standard rates worth passing a signal through, hardest first.

    Only rates below the source rate impose a band limit, so those are the
    real attacks and are ordered by how much they remove. One rate above the
    source is appended when available: upsampling and returning is a pure
    interpolation round trip, which is the mildest meaningful case.
    """
    lower = sorted((rate for rate in STANDARD_RATES if rate < sample_rate), reverse=True)
    higher = sorted(rate for rate in STANDARD_RATES if rate > sample_rate)
    targets = lower
    if higher:
        targets = targets + [higher[0]]
    return targets or [sample_rate]


def _resample_parameters(severity: Severity, sample_rate: int) -> dict[str, Any]:
    """Pick a resampling target for a severity, given the source rate."""
    targets = resampling_targets(sample_rate)
    lower = [rate for rate in targets if rate < sample_rate]
    higher = [rate for rate in targets if rate > sample_rate]

    if severity is Severity.MILD:
        # Prefer the interpolation-only round trip; fall back to the gentlest
        # downsampling when the source is already at the top of the ladder.
        choice = higher[0] if higher else lower[0]
    elif not lower:
        choice = higher[0] if higher else sample_rate
    elif severity is Severity.MODERATE:
        choice = lower[0]
    elif severity is Severity.STRONG:
        choice = lower[min(1, len(lower) - 1)]
    else:
        choice = lower[-1]

    return {"intermediate_hz": int(choice)}


def severity_parameters(
    name: str, severity: Severity, sample_rate: int | None = None
) -> dict[str, Any]:
    """Concrete parameters for one attack at one severity.

    Args:
        name: Registered attack name.
        severity: Level to resolve.
        sample_rate: Needed by the families whose parameters depend on it
            (filtering, resampling); defaults to 16 kHz, the rate of the
            packaged speech datasets.
    """
    if not isinstance(severity, Severity):
        severity = Severity(str(severity).lower())

    rate = int(sample_rate or DEFAULT_SAMPLE_RATE)
    nyquist = rate / 2.0

    if name in {"awgn", "pink_noise"}:
        return {"snr_db": _NOISE_SNR_DB[severity]}
    if name == "impulse_noise":
        return {"snr_db": _NOISE_SNR_DB[severity], "density": _DROPOUT_FRACTION[severity]}
    if name in _CODEC_BITRATE_KBPS:
        return {"bitrate_kbps": _CODEC_BITRATE_KBPS[name][severity]}
    if name == "codec":
        return {"bitrate_kbps": _CODEC_BITRATE_KBPS["mp3"][severity]}
    if name == "low_pass":
        return {"cutoff_hz": round(_LOW_PASS_NYQUIST_FRACTION[severity] * nyquist, 1)}
    if name == "high_pass":
        cutoff = _HIGH_PASS_HZ[severity]
        if cutoff >= nyquist:
            cutoff = round(0.4 * nyquist, 1)
        return {"cutoff_hz": cutoff}
    if name == "band_pass":
        return {
            "low_hz": _HIGH_PASS_HZ[severity],
            "high_hz": round(_LOW_PASS_NYQUIST_FRACTION[severity] * nyquist, 1),
        }
    if name == "notch":
        return {
            "center_hz": round(0.25 * nyquist, 1),
            "quality": _NOTCH_QUALITY[severity],
        }
    if name == "smoothing":
        return {"window_length": _SMOOTHING_WINDOW[severity]}
    if name == "resample":
        return _resample_parameters(severity, rate)
    if name == "clock_drift":
        return {"offset_ppm": _CLOCK_PPM[severity]}
    if name == "bit_depth":
        return {"bits": _BIT_DEPTH[severity]}
    if name == "gain":
        return {"gain_db": _GAIN_DB[severity]}
    if name == "clipping":
        return {"threshold": _CLIPPING_THRESHOLD[severity], "mode": "peak"}
    if name == "compression_dynamic":
        return {"ratio": _COMPRESSION_RATIO[severity]}
    if name == "time_shift":
        return {"shift_ms": _SHIFT_MS[severity], "shift_samples": None}
    if name == "crop":
        return {"fraction": _CROP_FRACTION[severity]}
    if name == "dropout":
        return {"fraction": _DROPOUT_FRACTION[severity]}
    if name == "sample_jitter":
        return {"events": _JITTER_EVENTS[severity]}
    if name in {"time_stretch", "speed"}:
        return {"rate": _RATE_FACTOR[severity]}
    if name == "pitch_shift":
        return {"semitones": _PITCH_SEMITONES[severity]}
    if name == "echo":
        delay, attenuation = _ECHO[severity]
        return {"delay_ms": delay, "attenuation": attenuation}
    if name == "reverb":
        return {"rt60_seconds": _RT60[severity]}
    if name == "acoustic_channel":
        return {
            "rt60_seconds": _RT60[severity],
            "snr_db": _NOISE_SNR_DB[severity],
            "clock_offset_ppm": _CLOCK_PPM[severity] if severity is Severity.EXTREME else 0.0,
        }

    raise AttackError(f"no severity preset defined for attack {name!r}")


def severity_sweep(name: str, sample_rate: int | None = None) -> list[Attack]:
    """The same attack at all four severities."""
    return [
        create(name, **severity_parameters(name, level, sample_rate)) for level in Severity
    ]


#: Named channels made of several stages. Each one models a path audio really
#: takes, and the stage order is the order the operations happen in reality.
PIPELINES: dict[str, list[str]] = {
    # Upload to a streaming platform: transcode, loudness processing, and a
    # second transcode at the delivery bitrate.
    "streaming_upload": ["aac:bitrate_kbps=128", "compression_dynamic:ratio=4.0", "mp3:bitrate_kbps=128"],
    # Voice call: band limitation, low bitrate speech codec, packet loss.
    "voice_call": ["band_pass:low_hz=300,high_hz=3400", "opus:bitrate_kbps=32", "dropout:fraction=0.005"],
    # Broadcast chain: light compression, resampling to the delivery rate, noise.
    "broadcast": ["compression_dynamic:ratio=2.0", "resample", "awgn:snr_db=30"],
    # Playback and recapture in a room, then stored as a compressed file.
    "over_the_air": ["acoustic_channel", "aac:bitrate_kbps=96"],
    # An attacker deliberately desynchronising before re-encoding.
    "desync_attack": ["time_shift:shift_ms=10", "speed:rate=1.01", "mp3:bitrate_kbps=128"],
}


def benchmark_pipeline(name: str, sample_rate: int | None = None) -> AttackPipeline:
    """Build one of the named channel pipelines."""
    if name not in PIPELINES:
        raise AttackError(f"unknown pipeline {name!r}; known: {sorted(PIPELINES)}")
    stages = [build(spec, sample_rate) for spec in PIPELINES[name]]
    return chain(*stages, label=name)


def _codec_specs(codec: str, bitrates: Sequence[int]) -> list[str]:
    return [f"{codec}:bitrate_kbps={rate}" for rate in bitrates]


def quick_suite(sample_rate: int | None = None) -> list[str]:
    """Smoke test: one attack from each family, moderate severity.

    Deliberately small and fast; suitable for checking that a pipeline runs,
    not for drawing conclusions.
    """
    targets = resampling_targets(int(sample_rate or DEFAULT_SAMPLE_RATE))
    return [
        "awgn@moderate",
        "mp3:bitrate_kbps=128",
        "low_pass@moderate",
        f"resample:intermediate_hz={targets[0]}",
        "bit_depth:bits=8",
        "gain:gain_db=-6.0",
        "crop:fraction=0.01",
        "time_shift:shift_ms=10",
        "echo@moderate",
    ]


def standard_suite(sample_rate: int | None = None) -> list[str]:
    """The default scientific benchmark: every family at several levels.

    Covers the manipulations audio actually undergoes, at severities spanning
    "inaudible" to "clearly damaged", so a method's failure point can be
    located rather than merely observed.
    """
    rate = int(sample_rate or DEFAULT_SAMPLE_RATE)
    nyquist = rate / 2.0

    specs: list[str] = []
    specs += _codec_specs("mp3", (192, 128, 96, 64))
    specs += _codec_specs("aac", (192, 128, 96, 64))
    specs += _codec_specs("opus", (96, 64, 32))
    specs += [f"awgn:snr_db={snr}" for snr in (40, 30, 20, 15, 10)]
    specs += [f"pink_noise:snr_db={snr}" for snr in (30, 20)]
    specs += [f"resample:intermediate_hz={target}" for target in resampling_targets(rate)]
    specs += [f"bit_depth:bits={bits}" for bits in (16, 12, 8)]
    specs += [f"gain:gain_db={gain}" for gain in (-6.0, -3.0, 3.0, 6.0)]
    specs += [
        f"low_pass:cutoff_hz={round(fraction * nyquist, 1)}"
        for fraction in (0.9, 0.5, 0.3, 0.15)
    ]
    specs += [f"high_pass:cutoff_hz={cutoff}" for cutoff in (100.0, 300.0) if cutoff < nyquist]
    specs += [f"crop:fraction={fraction}" for fraction in (0.001, 0.01, 0.05)]
    specs += [f"time_shift:shift_ms={shift}" for shift in (1.0, 10.0, 100.0)]
    specs += [f"time_stretch:rate={rate_factor}" for rate_factor in (0.99, 1.01, 0.95, 1.05)]
    specs += [f"speed:rate={rate_factor}" for rate_factor in (0.99, 1.01)]
    specs += [
        f"echo:delay_ms={delay},attenuation={attenuation}"
        for delay, attenuation in ((5.0, 0.1), (25.0, 0.25), (50.0, 0.5))
    ]
    specs += ["clipping:threshold=0.9,mode=peak", "dropout:fraction=0.005", "reverb:rt60_seconds=0.5"]
    return specs


def full_suite(sample_rate: int | None = None) -> list[str]:
    """Dense sweep: every attack at every severity, plus the pipelines.

    Intended for a research run rather than routine use; it is large and
    includes the codec and acoustic attacks, which are the slow ones.
    """
    rate = int(sample_rate or DEFAULT_SAMPLE_RATE)
    specs = list(standard_suite(rate))

    for name in (
        "awgn", "pink_noise", "impulse_noise", "low_pass", "high_pass", "band_pass", "notch",
        "smoothing", "resample", "clock_drift", "bit_depth", "gain", "clipping",
        "compression_dynamic", "time_shift", "crop", "dropout", "sample_jitter",
        "time_stretch", "speed", "pitch_shift", "echo", "reverb", "acoustic_channel",
        "mp3", "aac", "opus", "vorbis",
    ):
        specs += [f"{name}@{level.value}" for level in Severity]

    specs += [f"pipeline:name={name}" for name in PIPELINES]
    return list(dict.fromkeys(specs))


SUITES = {
    "quick": quick_suite,
    "standard": standard_suite,
    "full": full_suite,
}


def benchmark_suite(preset: str = "standard", sample_rate: int | None = None) -> list[str]:
    """Attack specifications for a named benchmark preset."""
    key = str(preset).lower()
    if key not in SUITES:
        raise AttackError(f"unknown benchmark preset {preset!r}; known: {sorted(SUITES)}")
    return SUITES[key](sample_rate)


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "PIPELINES",
    "STANDARD_RATES",
    "SUITES",
    "benchmark_pipeline",
    "benchmark_suite",
    "full_suite",
    "quick_suite",
    "resampling_targets",
    "severity_parameters",
    "severity_sweep",
    "standard_suite",
]
