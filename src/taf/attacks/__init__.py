"""Robustness attacks for audio steganography evaluation.

The package is organised by the phenomenon each attack models:

* :mod:`taf.attacks.codec` - real lossy encoders (MP3, AAC, Opus, Vorbis)
* :mod:`taf.attacks.noise` - additive white, pink and impulsive noise at a target SNR
* :mod:`taf.attacks.filtering` - low/high/band-pass, notch and smoothing
* :mod:`taf.attacks.resampling` - sample-rate round trips and clock drift
* :mod:`taf.attacks.quantization` - PCM bit-depth reduction
* :mod:`taf.attacks.amplitude` - gain, clipping and dynamic-range compression
* :mod:`taf.attacks.temporal` - shift, crop, jitter, dropout, stretch, speed, pitch
* :mod:`taf.attacks.acoustic` - echo, reverberation and a simulated acoustic path
* :mod:`taf.attacks.pipeline` - several attacks composed into one channel
* :mod:`taf.attacks.presets` - severity levels and benchmark suites

Every attack returns an :class:`~taf.attacks.base.AttackResult` carrying the
processed audio and the parameters that produced it, which is what makes a
benchmark row reproducible.
"""
from taf.attacks.acoustic import AcousticChannel, EchoAttack, Reverberation
from taf.attacks.amplitude import Clipping, DynamicRangeCompression, GainChange
from taf.attacks.attacks import CorruptedWavFile
from taf.attacks.base import (
    Attack,
    AttackCategory,
    AttackError,
    AttackResult,
    AttackToolUnavailableError,
    Severity,
)
from taf.attacks.codec import CodecCompression, ffmpeg_available
from taf.attacks.filtering import (
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
    MovingAverageSmoothing,
    NotchFilter,
)
from taf.attacks.noise import AdditivePinkNoise, AdditiveWhiteNoise, ImpulseNoise
from taf.attacks.pipeline import AttackPipeline, chain
from taf.attacks.presets import (
    benchmark_pipeline,
    benchmark_suite,
    severity_parameters,
    severity_sweep,
)
from taf.attacks.quantization import BitDepthReduction
from taf.attacks.registry import (
    LEGACY_ALIASES,
    available_attacks,
    build,
    build_all,
    create,
    resolve_name,
)
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
from taf.models.WavFile import WavFile


def apply_all_attacks(wav_file: WavFile) -> CorruptedWavFile:
    """A representative chain across the attack families.

    Kept for compatibility with the original helper. It is a demonstration,
    not a benchmark: for evaluation use ``benchmark_suite()``, which applies
    each attack separately so their effects can be told apart.
    """
    return (
        CorruptedWavFile(wav_file)
        .low_pass_filter()
        .resample()
        .additive_noise()
        .gain()
        .crop()
        .time_shift()
        .echo_addition()
    )


__all__ = [
    "AcousticChannel",
    "AdditivePinkNoise",
    "AdditiveWhiteNoise",
    "Attack",
    "AttackCategory",
    "AttackError",
    "AttackPipeline",
    "AttackResult",
    "AttackToolUnavailableError",
    "BandPassFilter",
    "BitDepthReduction",
    "Clipping",
    "CodecCompression",
    "CorruptedWavFile",
    "Cropping",
    "DynamicRangeCompression",
    "EchoAttack",
    "GainChange",
    "HighPassFilter",
    "ImpulseNoise",
    "LowPassFilter",
    "MovingAverageSmoothing",
    "NotchFilter",
    "PitchShift",
    "ResampleRoundTrip",
    "Reverberation",
    "SampleDropout",
    "SampleInsertionDeletion",
    "SampleRateOffset",
    "Severity",
    "SpeedChange",
    "TimeShift",
    "TimeStretch",
    "ZeroPadding",
    "LEGACY_ALIASES",
    "resolve_name",
    "apply_all_attacks",
    "available_attacks",
    "benchmark_pipeline",
    "benchmark_suite",
    "build",
    "build_all",
    "chain",
    "create",
    "ffmpeg_available",
    "severity_parameters",
    "severity_sweep",
]
