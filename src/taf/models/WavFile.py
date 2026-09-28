from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class WavFile:
    samplerate: int
    samples: np.ndarray
    path: Path
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def load(path: Path) -> WavFile:
        from taf.audio.metadata import describe_audio

        samples, fs = sf.read(path, dtype='float32')
        return WavFile(samplerate=fs, samples=samples, path=path, metadata=describe_audio(path))
