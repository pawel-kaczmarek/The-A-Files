from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from taf.audio.formats import AudioFileFormat
from taf.audio.io import AudioCodecUnavailableError, load_audio, save_audio
from taf.models.WavFile import WavFile


def _roundtrip(samples: np.ndarray, samplerate: int, fmt: AudioFileFormat, tmp_path: Path) -> WavFile:
    target = tmp_path / f"signal{fmt.extension}"
    wav = WavFile(samplerate=samplerate, samples=samples, path=target)
    saved = save_audio(wav, target, fmt)
    assert saved == target
    assert saved.exists()
    return load_audio(saved)


def test_wav_roundtrip_preserves_samples(
    synthetic_sine: np.ndarray, sample_rate: int, tmp_path: Path
) -> None:
    loaded = _roundtrip(synthetic_sine, sample_rate, AudioFileFormat.WAV, tmp_path)
    assert loaded.samplerate == sample_rate
    assert loaded.samples.shape == synthetic_sine.shape
    # Default WAV subtype is FLOAT (32-bit float) — roundtrip should be exact.
    np.testing.assert_allclose(loaded.samples, synthetic_sine, rtol=0, atol=1e-7)


def test_flac_roundtrip_preserves_samples_within_pcm16_tolerance(
    synthetic_sine: np.ndarray, sample_rate: int, tmp_path: Path
) -> None:
    if "FLAC" not in sf.available_formats():
        pytest.skip("libsndfile in this environment cannot write FLAC")

    loaded = _roundtrip(synthetic_sine, sample_rate, AudioFileFormat.FLAC, tmp_path)
    assert loaded.samplerate == sample_rate
    assert loaded.samples.shape == synthetic_sine.shape
    # Default FLAC subtype is PCM_16 — within one LSB of 16-bit quantization.
    np.testing.assert_allclose(loaded.samples, synthetic_sine, rtol=0, atol=1.0 / 2**15)


def test_save_audio_raises_for_unsupported_codec(
    synthetic_sine: np.ndarray, sample_rate: int, tmp_path: Path
) -> None:
    wav = WavFile(samplerate=sample_rate, samples=synthetic_sine, path=tmp_path / "x.mp3")
    with pytest.raises(AudioCodecUnavailableError):
        save_audio(wav, tmp_path / "x.mp3", AudioFileFormat.MP3)
