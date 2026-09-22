"""Signal manipulations used to test watermark robustness.

The compression, filtering, speed and suppression attacks follow the
benchmark set of Liu et al., "AudioMarkBench: Benchmarking Robustness of
Audio Watermarking", NeurIPS 2024 Datasets & Benchmarks
(https://arxiv.org/abs/2406.06979), so results here can be read next to the
numbers reported there.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import fftfreq
from scipy.signal import butter, sosfilt

from taf.models.WavFile import WavFile


class AttackToolUnavailableError(RuntimeError):
    """Raised when an attack needs an external tool that is not installed."""


@dataclass
class CorruptedWavFile(WavFile):

    def __init__(self, wav_file: WavFile):
        self.samples = wav_file.samples
        self.samplerate = wav_file.samplerate
        self.path = wav_file.path

    def low_pass_filter(self, order: int = 16, cutoff_freq: float = 2000) -> CorruptedWavFile:
        sos = butter(order, cutoff_freq, fs=self.samplerate, output='sos', btype='low')
        self.samples = sosfilt(sos, self.samples)
        return self

    def additive_noise(self, std: float = 0.001) -> CorruptedWavFile:
        self.samples = np.random.normal(0, std, self.samples.shape[0]) + self.samples
        return self

    def frequency_filter(self, cutoff_frequency=2000) -> CorruptedWavFile:
        W = np.fft.fftshift(fftfreq(len(self.samples), d=1 / self.samplerate))
        fft = np.fft.fftshift(np.fft.fft(self.samples))
        filtered_fft = fft.copy()
        filtered_fft[(np.abs(W) == cutoff_frequency)] = 0
        self.samples = np.fft.ifft(np.fft.ifftshift(filtered_fft)).real
        return self

    def flip_random_samples(self, samples_to_flip=200) -> CorruptedWavFile:
        to_flip = np.random.choice(len(self.samples), samples_to_flip, replace=False)
        data = self.samples.copy()
        data[to_flip] = -data[to_flip]
        self.samples = data
        return self

    def cut_random_samples(self, samples_to_cut=200) -> CorruptedWavFile:
        to_flip = np.random.choice(len(self.samples), samples_to_cut, replace=False)
        data = self.samples.copy()
        data[to_flip] = 0
        self.samples = data
        return self

    def resample(self, target_samplerate: int = 27500) -> CorruptedWavFile:
        self.samples = librosa.resample(
            self.samples, orig_sr=self.samplerate, target_sr=target_samplerate, scale=True
        )
        self.samplerate = target_samplerate
        return self

    def amplitude_scaling(self, scale: float = 1.1) -> CorruptedWavFile:
        self.samples = np.multiply(self.samples, scale)
        return self

    def pitch_shift(self, n_steps: int = 4, bins_per_octave: int = 12) -> CorruptedWavFile:
        self.samples = librosa.effects.pitch_shift(
            self.samples, sr=self.samplerate, n_steps=n_steps, bins_per_octave=bins_per_octave
        )
        return self

    def time_stretch(self, rate: float = 2.0) -> CorruptedWavFile:
        self.samples = librosa.effects.time_stretch(self.samples, rate=rate)
        return self

    def high_pass_filter(self, order: int = 16, cutoff_freq: float = 500) -> CorruptedWavFile:
        """Remove low frequencies, where much watermark energy tends to sit."""
        sos = butter(order, cutoff_freq, fs=self.samplerate, output='sos', btype='high')
        self.samples = sosfilt(sos, self.samples)
        return self

    def quantization(self, bit_depth: int = 8) -> CorruptedWavFile:
        """Requantise to a coarser bit depth."""
        if not 2 <= bit_depth <= 16:
            raise ValueError("bit_depth must be in [2, 16]")

        levels = 2 ** (bit_depth - 1)
        self.samples = np.round(np.clip(self.samples, -1.0, 1.0) * levels) / levels
        return self

    def smoothing(self, window_length: int = 5) -> CorruptedWavFile:
        """Moving-average filter; erases fine sample-level detail."""
        if window_length < 2:
            raise ValueError("window_length must be at least 2")

        window = np.ones(window_length) / window_length
        self.samples = np.convolve(self.samples, window, mode='same')
        return self

    def echo_addition(self, delay_seconds: float = 0.1, decay: float = 0.3) -> CorruptedWavFile:
        """Add a plain echo, which confuses echo-hiding detectors."""
        delay = int(delay_seconds * self.samplerate)
        if delay < 1 or delay >= len(self.samples):
            raise ValueError("delay must be shorter than the signal")

        echoed = self.samples.copy()
        echoed[delay:] += decay * self.samples[:-delay]
        self.samples = echoed
        return self

    def speed_change(self, rate: float = 1.05) -> CorruptedWavFile:
        """Resample without pitch correction: playback runs faster or slower.

        Unlike time_stretch, this keeps the waveform shape and only reindexes
        it, which is the attack histogram-based methods are built to survive.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")

        self.samples = librosa.resample(
            self.samples, orig_sr=self.samplerate, target_sr=int(self.samplerate / rate)
        )
        return self

    def crop(self, fraction: float = 0.1) -> CorruptedWavFile:
        """Cut a fraction of the signal away, split between the two ends."""
        if not 0 < fraction < 1:
            raise ValueError("fraction must be in (0, 1)")

        margin = int(len(self.samples) * fraction / 2)
        self.samples = self.samples[margin:len(self.samples) - margin]
        return self

    def zero_padding(self, fraction: float = 0.1) -> CorruptedWavFile:
        """Prepend silence, which desynchronises position-based decoders."""
        if fraction <= 0:
            raise ValueError("fraction must be positive")

        pad = np.zeros(int(len(self.samples) * fraction))
        self.samples = np.concatenate((pad, self.samples))
        return self

    def sample_suppression(self, fraction: float = 0.01, run_length: int = 20) -> CorruptedWavFile:
        """Zero out short runs of samples scattered through the signal."""
        if not 0 < fraction < 1:
            raise ValueError("fraction must be in (0, 1)")

        data = self.samples.copy()
        runs = max(1, int(len(data) * fraction / run_length))
        starts = np.random.choice(max(1, len(data) - run_length), runs, replace=False)
        for start in starts:
            data[start:start + run_length] = 0.0
        self.samples = data
        return self

    def mp3_compression(self, bitrate: str = "64k") -> CorruptedWavFile:
        """Round-trip through MP3."""
        self.samples = self._transcode("mp3", ["-c:a", "libmp3lame", "-b:a", bitrate])
        return self

    def aac_compression(self, bitrate: str = "64k") -> CorruptedWavFile:
        """Round-trip through AAC."""
        self.samples = self._transcode("m4a", ["-c:a", "aac", "-b:a", bitrate])
        return self

    def opus_compression(self, bitrate: str = "32k") -> CorruptedWavFile:
        """Round-trip through Opus."""
        self.samples = self._transcode("opus", ["-c:a", "libopus", "-b:a", bitrate])
        return self

    def _transcode(self, suffix: str, codec_options: list[str]) -> np.ndarray:
        """Encode to a lossy format with ffmpeg and read the result back.

        The decoded signal is trimmed or padded to the original length:
        encoder delay otherwise shifts every sample, and that shift would be
        measured as robustness loss the codec did not actually cause.
        """
        if shutil.which("ffmpeg") is None:
            raise AttackToolUnavailableError(
                "ffmpeg is required for compression attacks but was not found on PATH"
            )

        sample_count = len(self.samples)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.wav"
            compressed = Path(directory) / f"out.{suffix}"
            restored = Path(directory) / "out.wav"

            sf.write(str(source), self.samples, self.samplerate, subtype="PCM_16")
            for command in (
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), *codec_options,
                 str(compressed)],
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(compressed),
                 "-acodec", "pcm_s16le", "-ar", str(self.samplerate), str(restored)],
            ):
                subprocess.run(command, check=True, capture_output=True)

            decoded, _ = sf.read(str(restored), dtype="float32")

        if decoded.ndim > 1:
            decoded = decoded[:, 0]
        if len(decoded) >= sample_count:
            return decoded[:sample_count]
        return np.pad(decoded, (0, sample_count - len(decoded)))
