from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from taf.audio.formats import AudioFileFormat
from taf.models.WavFile import WavFile


class AudioCodecUnavailableError(RuntimeError):
    pass


_SOUNDFILE_FORMATS: dict[AudioFileFormat, str] = {
    AudioFileFormat.WAV: "WAV",
    AudioFileFormat.FLAC: "FLAC",
    AudioFileFormat.OGG: "OGG",
}

_DEFAULT_SUBTYPES: dict[AudioFileFormat, str] = {
    AudioFileFormat.WAV: "FLOAT",
    AudioFileFormat.FLAC: "PCM_16",
}


def load_audio(path: str | Path) -> WavFile:
    return WavFile.load(Path(path))


def save_audio(
        wav_file: WavFile,
        path: str | Path,
        audio_format: AudioFileFormat | str,
        codec_options: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    file_format = AudioFileFormat.from_value(audio_format)
    options = codec_options or {}

    if file_format not in _SOUNDFILE_FORMATS:
        raise AudioCodecUnavailableError(
            f"Saving {file_format.value.upper()} requires an optional codec backend."
        )

    sf_format = _SOUNDFILE_FORMATS[file_format]
    if sf_format not in sf.available_formats():
        raise AudioCodecUnavailableError(
            f"libsndfile does not support writing {file_format.value.upper()} in this environment."
        )

    subtype = options.get("subtype", _DEFAULT_SUBTYPES.get(file_format))
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        file=target,
        data=wav_file.samples,
        samplerate=wav_file.samplerate,
        format=sf_format,
        subtype=subtype,
    )
    return target
