"""Download a corpus and prepare a reproducible evaluation subset.

Evaluating on a whole corpus is rarely needed and rarely affordable, but an
ad-hoc subset makes results impossible to compare. A prepared subset is
therefore defined by a rule, not by hand:

* candidate files are the archive members matching the corpus glob (and path
  filter), ordered by name;
* files outside the duration bounds are dropped;
* the selection is drawn with a seeded generator, round-robin over speakers
  when the corpus has them, so that no speaker dominates;
* every selected file is converted to mono and resampled to the target rate
  with a polyphase filter, optionally cut to a fixed-length excerpt, and
  written as 16-bit FLAC. Levels are left as recorded.

The resulting ``manifest.json`` records the archive digest, the rule and the
SHA-256 of every prepared file; preparing again with the same rule and
archive yields the same files.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from taf.corpora.catalog import Corpus

Progress = Callable[[str, float], None]
"""``progress(stage, fraction)``: stage is "download", "scan" or "prepare"."""

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")


@dataclass(frozen=True)
class SubsetRule:
    max_files: int = 100
    target_sample_rate: int | None = 16000
    min_duration_seconds: float = 1.0
    max_duration_seconds: float | None = 30.0
    #: Cut every file to this many seconds (after ``excerpt_offset_seconds``).
    excerpt_seconds: float | None = None
    excerpt_offset_seconds: float = 0.0
    seed: int = 0


def _noop(stage: str, fraction: float) -> None:
    return None


def download(url: str, destination: Path, progress: Progress = _noop, chunk_size: int = 1 << 20) -> Path:
    """Stream ``url`` to ``destination``; an existing complete file is reused."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        progress("download", 1.0)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        received = 0
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
                received += len(chunk)
                if total:
                    progress("download", received / total)
    partial.replace(destination)
    progress("download", 1.0)
    return destination


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Archive access
# --------------------------------------------------------------------------


class _Source:
    """Uniform read access to a tar archive, a zip archive or a directory."""

    def __init__(self, location: Path) -> None:
        self.location = location
        name = location.name.lower()
        if location.is_dir():
            self.kind = "dir"
        elif name.endswith(".zip"):
            self.kind = "zip"
            self._zip = zipfile.ZipFile(location)
        elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
            self.kind = "tar"
            self._tar = tarfile.open(location)
            self._members = {m.name: m for m in self._tar.getmembers() if m.isfile()}
        else:
            raise ValueError(f"unsupported archive: {location}")

    def names(self) -> list[str]:
        if self.kind == "dir":
            return sorted(
                path.relative_to(self.location).as_posix()
                for path in self.location.rglob("*")
                if path.is_file()
            )
        if self.kind == "zip":
            return sorted(info.filename for info in self._zip.infolist() if not info.is_dir())
        return sorted(self._members)

    def read(self, name: str) -> bytes:
        if self.kind == "dir":
            return (self.location / name).read_bytes()
        if self.kind == "zip":
            return self._zip.read(name)
        extracted = self._tar.extractfile(self._members[name])
        assert extracted is not None
        return extracted.read()

    def close(self) -> None:
        if self.kind == "zip":
            self._zip.close()
        elif self.kind == "tar":
            self._tar.close()


def _matches(name: str, corpus: Corpus) -> bool:
    path = PurePosixPath(name)
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        return False
    # PurePosixPath.match anchors at the right, so "**/audio/*.wav" becomes
    # "audio/*.wav" and matches at any depth.
    pattern = corpus.audio_glob.removeprefix("**/")
    if not path.match(pattern):
        return False
    return corpus.path_filter is None or corpus.path_filter in name


def _speaker(name: str, corpus: Corpus) -> str | None:
    if corpus.speaker_level is None:
        return None
    parts = PurePosixPath(name).parts
    return parts[-1 - corpus.speaker_level] if len(parts) > corpus.speaker_level else None


# --------------------------------------------------------------------------
# Selection and conversion
# --------------------------------------------------------------------------


def _round_robin(names: list[str], corpus: Corpus, rng: np.random.Generator) -> Iterator[str]:
    """Candidates in random order, alternating between speakers."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(_speaker(name, corpus) or "", []).append(name)
    queues = []
    for speaker in sorted(groups):
        members = list(groups[speaker])
        rng.shuffle(members)
        queues.append(members)
    order = list(range(len(queues)))
    rng.shuffle(order)
    while any(queues):
        for index in order:
            if queues[index]:
                yield queues[index].pop()


def _to_target(samples: np.ndarray, rate: int, rule: SubsetRule) -> tuple[np.ndarray, int]:
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    target = rule.target_sample_rate or rate
    if target != rate:
        ratio = Fraction(target, rate).limit_denominator(1000)
        mono = resample_poly(mono, ratio.numerator, ratio.denominator)
    if rule.excerpt_seconds:
        start = int(rule.excerpt_offset_seconds * target)
        mono = mono[start : start + int(rule.excerpt_seconds * target)]
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        # Resampling can overshoot a full-scale input; scale down rather than clip.
        mono = mono / peak
    return mono.astype(np.float64), target


def prepare_subset(
    corpus: Corpus,
    source: Path,
    destination: Path,
    rule: SubsetRule,
    progress: Progress = _noop,
) -> dict:
    """Select, convert and write a subset of ``source`` into ``destination``."""
    reader = _Source(source)
    try:
        candidates = [name for name in reader.names() if _matches(name, corpus)]
        progress("scan", 1.0)
        if not candidates:
            raise ValueError(f"no audio files matching {corpus.audio_glob!r} in {source}")

        destination.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(rule.seed)
        files: list[dict] = []
        examined = 0
        for name in _round_robin(candidates, corpus, rng):
            if len(files) >= rule.max_files:
                break
            examined += 1
            try:
                samples, rate = sf.read(io.BytesIO(reader.read(name)), always_2d=False)
            except Exception:  # noqa: BLE001 - an unreadable member is skipped and counted
                continue
            duration = len(samples) / rate
            if rule.excerpt_seconds:
                if duration < rule.excerpt_offset_seconds + rule.excerpt_seconds:
                    continue
            elif duration < rule.min_duration_seconds or (
                rule.max_duration_seconds and duration > rule.max_duration_seconds
            ):
                continue
            audio, target = _to_target(np.asarray(samples, dtype=np.float64), rate, rule)
            stem = PurePosixPath(name).stem
            speaker = _speaker(name, corpus)
            if corpus.id == "musdb18hq":
                stem = PurePosixPath(name).parent.name
            output = destination / f"{len(files):04d}_{_safe(stem)}.flac"
            sf.write(output, audio, target, subtype="PCM_16")
            files.append(
                {
                    "file": output.name,
                    "source": name,
                    "speaker": speaker,
                    "sample_rate": target,
                    "duration_seconds": round(len(audio) / target, 4),
                    "sha256": sha256_of(output),
                }
            )
            progress("prepare", len(files) / rule.max_files)
    finally:
        reader.close()

    manifest = {
        "corpus": corpus.id,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source.name),
        "source_sha256": sha256_of(source) if source.is_file() else None,
        "rule": asdict(rule),
        "candidates": len(candidates),
        "examined": examined,
        "speakers": len({f["speaker"] for f in files if f["speaker"]}),
        "total_duration_seconds": round(sum(f["duration_seconds"] for f in files), 3),
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _safe(stem: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:80]


def scan_directory(directory: Path) -> dict:
    """Describe an existing directory of audio files (a registered local dataset)."""
    files = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file():
            continue
        try:
            info = sf.info(str(path))
        except Exception:  # noqa: BLE001 - unreadable files are left out
            continue
        files.append(
            {
                "file": path.relative_to(directory).as_posix(),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_seconds": round(info.frames / info.samplerate, 4) if info.samplerate else None,
            }
        )
    return {
        "files": files,
        "total_duration_seconds": round(sum(f["duration_seconds"] or 0 for f in files), 3),
    }


__all__ = [
    "Progress",
    "SubsetRule",
    "download",
    "prepare_subset",
    "scan_directory",
    "sha256_of",
]
