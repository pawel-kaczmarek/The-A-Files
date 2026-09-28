"""Deterministic synthetic test signals.

Natural recordings rarely contain the conditions under which embedding
methods fail: a pure tone has a degenerate amplitude distribution, digital
silence has no energy to hide bits in, a full-scale square wave leaves no
headroom. A small set of synthetic signals exposes these edge cases. Every
signal is generated from its definition and a fixed seed at -20 dBFS RMS
(except where the definition says otherwise), so the set is identical on
every machine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import chirp, lfilter

from taf.corpora.prepare import sha256_of

TARGET_RMS_DBFS = -20.0


def _at_level(signal: np.ndarray, dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(signal**2)))
    if rms == 0:
        return signal
    scaled = signal * (10 ** (dbfs / 20) / rms)
    peak = float(np.max(np.abs(scaled)))
    return scaled / peak * 0.99 if peak >= 1.0 else scaled


def _pink(length: int, rng: np.random.Generator) -> np.ndarray:
    # Paul Kellet's economy filter: -3 dB/octave within 0.5 dB above 9 Hz.
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, rng.standard_normal(length))


def synthetic_signals(sample_rate: int = 16000, duration_seconds: float = 5.0, seed: int = 0) -> dict[str, np.ndarray]:
    """Name -> signal for every synthetic test condition."""
    rng = np.random.default_rng(seed)
    length = int(sample_rate * duration_seconds)
    t = np.arange(length) / sample_rate
    nyquist = sample_rate / 2
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t))  # syllabic rate, ~4 Hz

    bursts = np.zeros(length)
    period = sample_rate // 2
    for start in range(0, length, period):
        bursts[start : start + period // 4] = np.sin(2 * np.pi * 1000.0 * t[start : start + period // 4])

    return {
        "tone_250hz": _at_level(np.sin(2 * np.pi * 250.0 * t)),
        "tone_1khz": _at_level(np.sin(2 * np.pi * 1000.0 * t)),
        "tone_4khz": _at_level(np.sin(2 * np.pi * min(4000.0, nyquist * 0.8) * t)),
        "log_sweep": _at_level(chirp(t, f0=50.0, f1=nyquist * 0.9, t1=duration_seconds, method="logarithmic")),
        "white_noise": _at_level(rng.standard_normal(length)),
        "pink_noise": _at_level(_pink(length, rng)),
        "modulated_pink_noise": _at_level(_pink(length, rng) * envelope),
        "tone_bursts_in_silence": _at_level(bursts),
        "square_wave_full_scale": 0.99 * np.sign(np.sin(2 * np.pi * 200.0 * t)),
        "low_level_noise_-60dbfs": _at_level(rng.standard_normal(length), -60.0),
    }


def write_synthetic_set(destination: Path, sample_rate: int = 16000, duration_seconds: float = 5.0, seed: int = 0) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for index, (name, signal) in enumerate(synthetic_signals(sample_rate, duration_seconds, seed).items()):
        output = destination / f"{index:04d}_{name}.flac"
        sf.write(output, signal, sample_rate, subtype="PCM_16")
        files.append(
            {
                "file": output.name,
                "channels": 1,
                "bit_depth": 16,
                "subtype": "PCM_16",
                "category": "synthetic_signal",
                "source": name,
                "speaker": None,
                "sample_rate": sample_rate,
                "duration_seconds": len(signal) / sample_rate,
                "sha256": sha256_of(output),
            }
        )
    manifest = {
        "corpus": "synthetic",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "rule": {"sample_rate": sample_rate, "duration_seconds": duration_seconds, "seed": seed},
        "total_duration_seconds": duration_seconds * len(files),
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


__all__ = ["synthetic_signals", "write_synthetic_set"]
