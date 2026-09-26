"""Run manifest: what is needed to reproduce, or audit, an experiment.

A configuration alone does not pin a result. The same configuration can give
different numbers with another version of the package or of numpy/scipy, with
a different FFmpeg (every codec attack goes through it), or on audio files
that were silently replaced. The manifest records all of these next to the
resolved configuration, including the seed that was drawn when none was
given.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

#: Distributions whose version can change a result.
TRACKED_PACKAGES = (
    "the-a-files",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "PyWavelets",
    "pesq",
    "pystoi",
    "museval",
    "numba",
    "torch",
    "tensorflow",
    "audioseal",
    "wavmark",
)


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def ffmpeg_version() -> str | None:
    """First line of ``ffmpeg -version``, or ``None`` when FFmpeg is absent."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "-version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    return first_line.strip() or None


def source_revision() -> dict[str, Any] | None:
    """Git commit of the source tree the package runs from, if it is a checkout.

    ``dirty`` flags uncommitted changes, which make the commit alone
    insufficient to reproduce the run.
    """
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists() or shutil.which("git") is None:
        return None

    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status)}


def file_digest(path: str | Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def describe_inputs(files: Iterable[Any]) -> list[dict[str, Any]]:
    """Name, SHA-256, sampling rate and duration of every input file."""
    described: list[dict[str, Any]] = []
    for wav_file in files:
        path = Path(wav_file.path)
        samples = len(wav_file.samples)
        described.append(
            {
                "name": path.name,
                "sha256": file_digest(path),
                "sample_rate": wav_file.samplerate,
                "samples": samples,
                "duration_seconds": samples / wav_file.samplerate if wav_file.samplerate else None,
            }
        )
    return described


def build_manifest(config: Any, files: Iterable[Any], attacks: list[str]) -> dict[str, Any]:
    """Manifest of one run; ``config`` is the resolved ``ExperimentConfig``."""
    from taf import __version__

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taf_version": __version__,
        "source": source_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "ffmpeg": ffmpeg_version(),
        "random_seed": config.random_seed,
        "resolved_attacks": attacks,
        "max_workers": config.max_workers,
        # Timings from concurrent trials share the CPU and are not comparable.
        "timing_reliable": config.max_workers == 1,
        "inputs": describe_inputs(files),
        "config": config.model_dump(mode="json"),
    }


__all__ = [
    "TRACKED_PACKAGES",
    "build_manifest",
    "describe_inputs",
    "ffmpeg_version",
    "file_digest",
    "package_versions",
    "source_revision",
]
