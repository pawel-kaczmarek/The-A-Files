from pathlib import Path

from taf.models.WavFile import WavFile


class WavFileLoader:
    absolute_path: Path

    def __init__(self, absolute_path: Path):
        self.absolute_path = absolute_path

    def load(self, filename: str) -> WavFile:
        return WavFile.load(self.absolute_path / filename)
