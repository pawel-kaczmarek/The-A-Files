import os.path
from pathlib import Path

from TAF.models.WavFile import WavFile


class WavFileLoader:
    absolute_path: Path

    def __init__(self, absolute_path: Path):
        self.absolute_path = absolute_path

    def load(self, filename: str) -> WavFile:
        join = os.path.join(self.absolute_path, filename)
        return WavFile.load(Path(join))
