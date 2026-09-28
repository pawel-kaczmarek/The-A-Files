"""Container metadata, separate from the floating-point evaluation signal."""

import soundfile as sf


def describe_audio(path) -> dict:
    info = sf.info(path)
    # Lossy and float subtypes have no integer PCM bit depth.
    depths = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
    return {
        "sample_rate": info.samplerate, "channels": info.channels,
        "frames": info.frames, "duration_seconds": info.frames / info.samplerate,
        "format": info.format, "subtype": info.subtype,
        "bit_depth": depths.get(info.subtype),
    }
