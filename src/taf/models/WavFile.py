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

    def save_steganography_file(self, a_samples: np.ndarray, suffix: str | None = None):
        sf.write(file=self._steganography_filename(suffix), data=a_samples, samplerate=self.samplerate)

    def _steganography_filename(self, suffix: str | None = None):
        return "{0}_{2}{1}".format(self.path.stem, self.path.suffix, "_stego" + (suffix if suffix is not None else ""))
