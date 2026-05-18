from __future__ import annotations

from enum import Enum


class DecodeTarget(str, Enum):
    DIRECT = "direct"


class AudioFileFormat(str, Enum):
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    MP3 = "mp3"
    AAC = "aac"

    @property
    def extension(self) -> str:
        return f".{self.value}"

    @property
    def is_lossy(self) -> bool:
        return self in {
            AudioFileFormat.OGG,
            AudioFileFormat.MP3,
            AudioFileFormat.AAC,
        }

    @classmethod
    def from_value(cls, value: AudioFileFormat | str) -> AudioFileFormat:
        if isinstance(value, AudioFileFormat):
            return value
        return cls(value.lower().lstrip("."))


def is_direct_target(value: AudioFileFormat | DecodeTarget | str) -> bool:
    if isinstance(value, DecodeTarget):
        return value == DecodeTarget.DIRECT
    if isinstance(value, AudioFileFormat):
        return False
    return value.lower().lstrip(".") == DecodeTarget.DIRECT.value
