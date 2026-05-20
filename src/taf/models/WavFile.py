from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class WavFile:
    samplerate: int
    samples: np.ndarray
    path: Path

    @staticmethod
    def load(path: Path) -> WavFile:
        samples, fs = sf.read(path, dtype='float32')
        return WavFile(samplerate=fs, samples=samples, path=path)
