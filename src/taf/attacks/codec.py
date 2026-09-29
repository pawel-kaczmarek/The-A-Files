"""Lossy codec attacks.

Perceptual codecs are the most realistic threat in this package: almost all
distributed audio has been through one. They are also the most informative,
because a codec is an adversary with a psychoacoustic model - it spends its
bits where the ear listens and discards what it judges inaudible, which is
precisely where a transparent watermark has hidden itself. The better the
embedding is perceptually, the more exactly it coincides with what the codec
throws away.

Real encoders are invoked through FFmpeg. Emulating a codec by low-passing or
rounding is not a substitute: those operations miss the block-based
transform, the adaptive quantisation driven by a masking model, the bit
reservoir and the joint-stereo decisions, all of which are what actually
destroy a watermark.

Because a real encoder is used, results depend on the FFmpeg build. The
encoder name and the FFmpeg version are recorded with every result.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from taf.attacks.base import (
    Attack,
    AttackCategory,
    AttackError,
    AttackToolUnavailableError,
)
from taf.models.card import AttackCard, Text

#: Container, FFmpeg encoder and typical bitrate span for each supported codec.
CODEC_SPECS: dict[str, dict[str, Any]] = {
    "mp3": {"suffix": "mp3", "encoder": "libmp3lame", "bitrates_kbps": (320, 192, 128, 96, 64)},
    "aac": {"suffix": "m4a", "encoder": "aac", "bitrates_kbps": (256, 192, 128, 96, 64)},
    "opus": {"suffix": "opus", "encoder": "libopus", "bitrates_kbps": (128, 96, 64, 32)},
    "vorbis": {"suffix": "ogg", "encoder": "libvorbis", "bitrates_kbps": (192, 128, 96, 64)},
}


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    """First line of ``ffmpeg -version``, or a marker when it is unavailable."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        return "unavailable"
    try:
        completed = subprocess.run(
            [executable, "-version"], check=True, capture_output=True, text=True
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return completed.stdout.splitlines()[0].strip() if completed.stdout else "unknown"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


#: Samples used to estimate the encoder delay. A few seconds is far more than
#: enough to lock onto a delay of at most a tenth of a second, and bounding it
#: keeps the correlation cheap regardless of file length.
_DELAY_WINDOW = 1 << 15


def _estimate_delay(reference: np.ndarray, decoded: np.ndarray, max_lag: int) -> int:
    """Lag, in samples, that best aligns ``decoded`` with ``reference``.

    Encoders prepend padding: LAME adds its own, and AAC encoders typically add
    over a thousand samples. That offset is a container artefact, not codec
    damage, and leaving it in makes every position-based method fail for the
    wrong reason.

    The correlation runs over a bounded prefix and through the FFT. A direct
    correlation over the whole signal is O(n^2) and took about 100 seconds per
    codec attack on 5 seconds of speech, which made a codec sweep impractical.
    """
    from scipy.signal import correlate

    window = min(len(reference), len(decoded), _DELAY_WINDOW)
    if window < 16:
        return 0

    a = np.asarray(reference[:window], dtype=np.float64)
    b = np.asarray(decoded[:window], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    if not np.any(a) or not np.any(b):
        return 0

    correlation = correlate(b, a, mode="full", method="fft")
    lags = np.arange(-window + 1, window)
    allowed = np.abs(lags) <= max_lag
    if not np.any(allowed):
        return 0

    return int(lags[allowed][int(np.argmax(correlation[allowed]))])


@dataclass(frozen=True)
class CodecCompression(Attack):
    """Encode with a real lossy codec and decode back to PCM.

    Args:
        codec: One of ``mp3``, ``aac``, ``opus``, ``vorbis``.
        bitrate_kbps: Constant bitrate target. Lower means fewer bits for the
            encoder to spend and a more aggressive attack.
        align_delay: Remove the encoder's padding by cross-correlation, so the
            measurement is of codec damage rather than of a constant offset.
            The estimated delay is always recorded, whether or not it is
            removed.
        restore_length: Trim or pad the decoded signal back to the input
            length, since codecs work in whole frames.
    """

    codec: str = "mp3"
    bitrate_kbps: int = 128
    align_delay: bool = True
    restore_length: bool = True

    name = "codec"
    category = AttackCategory.CODEC
    card = AttackCard(
        title=Text("Lossy codec round trip", "Kodowanie stratne"),
        summary=Text(
            en=(
                "Tests whether the message survives real lossy encoding and decoding (MP3, AAC, Opus or "
                "Vorbis)."
            ),
            pl=(
                "Sprawdza, czy wiadomość przetrwa rzeczywiste kodowanie i dekodowanie stratne (MP3, "
                "AAC, Opus lub Vorbis)."
            ),
        ),
        details=Text(
            en=(
                "Runs the codec through FFmpeg and returns decoded PCM. bitrate_kbps sets the target "
                "bitrate; a lower rate usually discards more information. Optional delay alignment and "
                "length restoration separate codec damage from timing changes. Perceptual compression "
                "can remove quiet embedded components; results depend on the encoder and FFmpeg build."
            ),
            pl=(
                "Uruchamia kodek przez FFmpeg i zwraca zdekodowany PCM. bitrate_kbps ustala docelową "
                "przepływność; niższa zwykle oznacza większą utratę informacji. Opcjonalne wyrównanie "
                "opóźnienia i przywrócenie długości oddzielają uszkodzenia kodeka od zmian czasowych. "
                "Kompresja percepcyjna może usuwać ciche składowe znaku; wynik zależy od enkodera i "
                "wersji FFmpeg."
            ),
        ),
    )

    def _process(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, dict[str, Any]]:
        codec = self.codec.lower()
        if codec not in CODEC_SPECS:
            raise AttackError(
                f"unknown codec {self.codec!r}; expected one of {sorted(CODEC_SPECS)}"
            )
        if self.bitrate_kbps <= 0:
            raise AttackError(f"bitrate_kbps must be positive, got {self.bitrate_kbps}")
        if not ffmpeg_available():
            raise AttackToolUnavailableError(
                "ffmpeg is required for codec attacks but was not found on PATH"
            )

        spec = CODEC_SPECS[codec]
        original_length = audio.shape[0]
        channels = 1 if audio.ndim == 1 else audio.shape[1]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.wav"
            compressed = Path(directory) / f"out.{spec['suffix']}"
            restored = Path(directory) / "out.wav"

            sf.write(str(source), audio, sample_rate, subtype="PCM_16")

            encode = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                "-c:a", spec["encoder"], "-b:a", f"{int(self.bitrate_kbps)}k",
                str(compressed),
            ]
            decode = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(compressed),
                "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels),
                str(restored),
            ]
            for command in (encode, decode):
                try:
                    subprocess.run(command, check=True, capture_output=True)
                except subprocess.CalledProcessError as error:
                    message = error.stderr.decode("utf-8", "replace").strip()
                    raise AttackError(f"ffmpeg failed for codec {codec}: {message}") from error

            # Evidence that a lossy file really was produced, rather than the
            # command silently passing PCM through.
            compressed_bytes = compressed.stat().st_size
            if compressed_bytes == 0:
                raise AttackError(f"codec {codec} produced an empty file")

            decoded, decoded_rate = sf.read(str(restored), dtype="float64", always_2d=(channels > 1))

        if decoded_rate != sample_rate:
            raise AttackError(
                f"decoder returned {decoded_rate} Hz, expected {sample_rate} Hz"
            )

        reference = audio if audio.ndim == 1 else audio[:, 0]
        probe = decoded if decoded.ndim == 1 else decoded[:, 0]
        delay = _estimate_delay(reference, probe, max_lag=sample_rate // 10)

        if self.align_delay and delay > 0:
            decoded = decoded[delay:]
        elif self.align_delay and delay < 0:
            pad_width = [(-delay, 0)] + [(0, 0)] * (decoded.ndim - 1)
            decoded = np.pad(decoded, pad_width)

        length_correction = 0
        if self.restore_length and decoded.shape[0] != original_length:
            length_correction = original_length - decoded.shape[0]
            if length_correction > 0:
                pad_width = [(0, length_correction)] + [(0, 0)] * (decoded.ndim - 1)
                decoded = np.pad(decoded, pad_width)
            else:
                decoded = decoded[:original_length]

        nominal_bits = original_length * channels * 16
        return decoded, sample_rate, {
            "codec": codec,
            "bitrate_kbps": int(self.bitrate_kbps),
            "encoder": spec["encoder"],
            "container": spec["suffix"],
            "ffmpeg_version": ffmpeg_version(),
            "compressed_bytes": int(compressed_bytes),
            "compression_ratio": float(nominal_bits / 8 / compressed_bytes),
            "encoder_delay_samples": int(delay),
            "delay_aligned": bool(self.align_delay),
            "length_correction_samples": int(length_correction),
        }


def mp3(bitrate_kbps: int = 128, **kwargs: Any) -> CodecCompression:
    return CodecCompression(codec="mp3", bitrate_kbps=bitrate_kbps, **kwargs)


def aac(bitrate_kbps: int = 128, **kwargs: Any) -> CodecCompression:
    return CodecCompression(codec="aac", bitrate_kbps=bitrate_kbps, **kwargs)


def opus(bitrate_kbps: int = 64, **kwargs: Any) -> CodecCompression:
    return CodecCompression(codec="opus", bitrate_kbps=bitrate_kbps, **kwargs)


def vorbis(bitrate_kbps: int = 128, **kwargs: Any) -> CodecCompression:
    return CodecCompression(codec="vorbis", bitrate_kbps=bitrate_kbps, **kwargs)


#: Display names of the codecs reachable through a registry shortcut.
CODEC_LABELS = {"mp3": "MP3", "aac": "AAC", "opus": "Opus", "vorbis": "Vorbis"}


def codec_shortcut_card(codec: str) -> AttackCard:
    """Card of a codec shortcut (``mp3``, ``aac``...): the codec card, named."""
    label = CODEC_LABELS[codec]
    return replace(
        CodecCompression.card,
        title=Text(f"{label} round trip", f"Kodowanie {label}"),
        summary=Text(
            en=f"Tests whether the message survives real {label} lossy encoding and decoding.",
            pl=f"Sprawdza, czy wiadomość przetrwa rzeczywiste kodowanie i dekodowanie stratne {label}.",
        ),
        abbreviation=label,
    )


__all__ = [
    "CODEC_LABELS",
    "CODEC_SPECS",
    "CodecCompression",
    "aac",
    "codec_shortcut_card",
    "ffmpeg_available",
    "ffmpeg_version",
    "mp3",
    "opus",
    "vorbis",
]
