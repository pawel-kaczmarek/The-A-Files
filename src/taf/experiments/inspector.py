"""Re-synthesis of a single trial for listening and inspection.

A result row records everything that determines its trial: the cover file,
the method with its parameters, the message bits, the attack specification
and - through the experiment seed, the file, the repetition and the attack -
the attack's random realisation. The signals of any trial can therefore be
rebuilt on demand instead of being stored. ``reproduced`` confirms that the
rebuilt trial decodes to the bits the run recorded; a method whose encoder is
not deterministic would show up here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from taf.experiments.results import ExperimentResultRow
from taf.experiments.schema import ExperimentConfig


@dataclass
class TrialAudio:
    sample_rate: int
    cover: np.ndarray
    stego: np.ndarray
    attacked: np.ndarray | None
    decoded_bits: str
    reproduced: bool
    attack_metadata: dict[str, Any]

    @property
    def residual(self) -> np.ndarray:
        """What embedding added: stego minus cover."""
        length = min(len(self.cover), len(self.stego))
        return self.stego[:length] - self.cover[:length]


def load_dataset_file(config: ExperimentConfig, file_name: str):
    """The single cover file ``file_name`` of a configuration's dataset."""
    from taf.audio.io import load_audio
    from taf.evaluation.workflow import audio_file_paths
    from taf.experiments.runner import _is_external, dataset_directory
    from taf.resources.paths import example_wav_path, packaged_dataset_audio_paths

    if _is_external(config):
        directory = dataset_directory(config)
        if directory is not None:
            for path in audio_file_paths(directory, recursive=True):
                if path.name == file_name:
                    return load_audio(path)
    elif config.dataset_id == "example":
        with example_wav_path() as path:
            return load_audio(path)
    else:
        with packaged_dataset_audio_paths() as groups:
            for path in (path for paths in groups.values() for path in paths):
                if Path(path).name == file_name:
                    return load_audio(path)
    raise FileNotFoundError(f"cover file {file_name!r} is not in the dataset of this run")


def resynthesize(config: ExperimentConfig, row: ExperimentResultRow) -> TrialAudio:
    from taf.attacks.registry import has_explicit_seed
    from taf.evaluation.seeding import attack_seed
    from taf.evaluation.workflow import _apply_attack
    from taf.plugins import create_method, format_method_spec

    if not row.message_bits or not row.method_type:
        raise ValueError("this row does not record enough to rebuild its trial")
    from taf.experiments.runner import load_dataset_files

    covers = load_dataset_files(config)
    matches = [file for file in covers if str(file.path) == row.file_path]
    if not matches:
        matches = [file for file in covers if file.metadata.get("file_id") == row.file_id]
    if len(matches) != 1:
        raise ValueError("The recorded cover is missing or ambiguous in the current dataset.")
    cover = matches[0]
    if row.audio_sha256 and cover.metadata.get("sha256") != row.audio_sha256:
        raise ValueError("Cover SHA-256 differs from the recorded experiment.")
    spec = format_method_spec(row.method_type, row.method_parameters)
    message = [int(bit) for bit in row.message_bits]

    stego = np.asarray(create_method(spec, cover.samplerate).encode(cover.samples.copy(), message), dtype=np.float64)
    attacked = None
    metadata: dict[str, Any] = {}
    signal = stego
    if row.attack:
        seed = None if has_explicit_seed(row.attack) else attack_seed(
            config.random_seed or 0, row.file_id or row.file_name, row.repetition, row.attack
        )
        attacked, _, metadata = _apply_attack(stego, cover.samplerate, row.attack, seed)
        attacked = np.asarray(attacked, dtype=np.float64)
        signal = attacked

    decoded = create_method(spec, cover.samplerate).decode(signal, len(message))
    decoded_bits = "".join(str(int(bit)) for bit in decoded)
    return TrialAudio(
        sample_rate=cover.samplerate,
        cover=np.asarray(cover.samples, dtype=np.float64),
        stego=stego,
        attacked=attacked,
        decoded_bits=decoded_bits,
        reproduced=decoded_bits == (row.decoded_bits or ""),
        attack_metadata=metadata,
    )


def spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    reference: float | None = None,
    n_fft: int = 512,
    max_frames: int = 320,
    max_bins: int = 128,
    floor_db: float = -100.0,
) -> dict[str, Any]:
    """Magnitude spectrogram in dB, reduced to at most ``max_frames`` x ``max_bins``.

    ``reference`` is the magnitude mapped to 0 dB; passing the cover's peak
    puts cover, stego and residual on one scale so they can be compared.
    """
    from scipy.signal import stft

    frequencies, times, values = stft(signal, fs=sample_rate, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    magnitude = np.abs(values)
    reference = reference or float(magnitude.max()) or 1.0
    frame_groups = np.array_split(np.arange(magnitude.shape[1]), min(max_frames, magnitude.shape[1]))
    bin_groups = np.array_split(np.arange(magnitude.shape[0]), min(max_bins, magnitude.shape[0]))
    reduced = np.array(
        [[magnitude[np.ix_(bins, frames)].mean() for frames in frame_groups] for bins in bin_groups]
    )
    db = np.maximum(20 * np.log10(np.maximum(reduced, 1e-12) / reference), floor_db)
    return {
        "times": [float(times[group].mean()) for group in frame_groups],
        "frequencies": [float(frequencies[group].mean()) for group in bin_groups],
        "db": np.round(db, 1).tolist(),
        "floor_db": floor_db,
    }


def peak_magnitude(signal: np.ndarray, sample_rate: int, n_fft: int = 512) -> float:
    from scipy.signal import stft

    _, _, values = stft(signal, fs=sample_rate, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    return float(np.abs(values).max()) or 1.0


def envelope(signal: np.ndarray, points: int = 800) -> dict[str, list[float]]:
    """Minimum and maximum per bucket, for drawing a waveform."""
    buckets = np.array_split(signal, min(points, max(1, len(signal))))
    return {
        "min": [float(bucket.min()) if bucket.size else 0.0 for bucket in buckets],
        "max": [float(bucket.max()) if bucket.size else 0.0 for bucket in buckets],
    }


def wav_bytes(signal: np.ndarray, sample_rate: int, normalise_to_dbfs: float | None = None) -> tuple[bytes, float]:
    """16-bit WAV of ``signal``; optionally scaled to a peak level, returning the gain in dB."""
    import io

    import soundfile as sf

    gain_db = 0.0
    data = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if normalise_to_dbfs is not None and peak > 0:
        target = 10 ** (normalise_to_dbfs / 20)
        gain_db = 20 * np.log10(target / peak)
        data = data * (target / peak)
    elif peak > 1.0:
        data = data / peak
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), gain_db


__all__ = [
    "TrialAudio",
    "envelope",
    "load_dataset_file",
    "peak_magnitude",
    "resynthesize",
    "spectrogram",
    "wav_bytes",
]
