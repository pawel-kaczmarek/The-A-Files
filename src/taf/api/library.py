"""The dataset library: corpora prepared, uploaded, registered or generated.

Preparation runs in the background: the record shows its stage (download,
scan, prepare) and progress while the archive is fetched and the subset is
written, and the finished record holds the manifest that pins the material.
Experiments name library datasets as ``library:<id>``; the resolver
registered with the engine maps that to the dataset directory.
"""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from taf.corpora import Corpus, SubsetRule, download, get_corpus, prepare_subset, scan_directory, write_synthetic_set
from taf.experiments.runner import register_dataset_resolver
from taf.experiments.schema import LIBRARY_DATASET_PREFIX
from taf.persistence import store
from taf.persistence.models import Dataset
from taf.persistence.settings import datasets_dir, downloads_dir

ALLOWED_UPLOAD_SUFFIXES = {".wav", ".flac", ".ogg"}
#: Library entries whose files the platform owns and deletes with them.
OWNED_KINDS = {"corpus", "upload", "synthetic"}


def _summary_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files") or []
    rates = sorted({entry.get("sample_rate") for entry in files if entry.get("sample_rate")})
    return {
        "file_count": len(files),
        "total_duration_seconds": manifest.get("total_duration_seconds"),
        "sample_rate": rates[0] if len(rates) == 1 else None,
    }


def resolve(key: str) -> Path | None:
    """Directory of the ready library dataset ``key``."""
    dataset_id = store.as_uuid(key)
    dataset = store.get_dataset(dataset_id) if dataset_id else None
    if dataset is None or dataset.status != "ready" or not dataset.path:
        return None
    return Path(dataset.path)


register_dataset_resolver(LIBRARY_DATASET_PREFIX, resolve)


class Library:
    def __init__(self) -> None:
        self._tasks: dict[uuid.UUID, asyncio.Task] = {}

    # ------------------------------------------------------------ corpora

    def prepare_corpus(
        self, corpus_id: str, name: str | None, rule: SubsetRule, source_path: str | None = None
    ) -> Dataset:
        corpus = get_corpus(corpus_id)
        if corpus.download is None and source_path is None:
            raise ValueError(
                f"{corpus.name} cannot be downloaded automatically ({corpus.license}); "
                "provide the path of a local copy."
            )
        if source_path is not None and not Path(source_path).exists():
            raise ValueError(f"Source not found: {source_path}")
        dataset = store.create_dataset(
            name=name or f"{corpus.name} ({rule.max_files} files, seed {rule.seed})",
            kind="corpus",
            corpus_id=corpus.id,
            status="pending",
            domain=corpus.domain,
            language=corpus.language,
            license=corpus.license,
            citation=corpus.citation,
            description=corpus.description,
            rule={**rule.__dict__, "source_path": source_path},
        )
        self._tasks[dataset.id] = asyncio.create_task(
            asyncio.to_thread(self._prepare, dataset.id, corpus, rule, source_path)
        )
        return dataset

    def _prepare(self, dataset_id: uuid.UUID, corpus: Corpus, rule: SubsetRule, source_path: str | None) -> None:
        last = {"time": 0.0}

        def progress(stage: str, fraction: float) -> None:
            now = time.monotonic()
            if now - last["time"] < 0.5 and fraction < 1.0:
                return
            last["time"] = now
            status = "downloading" if stage == "download" else "preparing"
            store.update_dataset(dataset_id, status=status, stage=stage, progress=round(fraction, 4))

        destination = datasets_dir() / str(dataset_id)
        try:
            if source_path is not None:
                source = Path(source_path)
            else:
                assert corpus.download is not None
                source = download(
                    corpus.download.url,
                    downloads_dir() / f"{corpus.id}.{corpus.download.archive}",
                    progress,
                )
            store.update_dataset(dataset_id, status="preparing", stage="scan", progress=0.0)
            manifest = prepare_subset(corpus, source, destination, rule, progress)
            store.update_dataset(
                dataset_id,
                status="ready",
                stage=None,
                progress=1.0,
                path=str(destination),
                manifest=manifest,
                **_summary_fields(manifest),
            )
        except Exception as error:  # noqa: BLE001 - recorded on the dataset
            logger.exception("Preparing {} failed: {}", corpus.id, error)
            shutil.rmtree(destination, ignore_errors=True)
            store.update_dataset(dataset_id, status="failed", error=str(error))

    # ------------------------------------------------------------ other sources

    def register_local(self, name: str, path: str, **metadata: Any) -> Dataset:
        directory = Path(path).expanduser()
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {path}")
        manifest = scan_directory(directory)
        if not manifest["files"]:
            raise ValueError(f"No audio files under {path}")
        corpus_id = metadata.pop("corpus_id", None)
        if corpus_id:
            corpus = get_corpus(corpus_id)
            for key in ("domain", "language", "license", "citation", "description"):
                metadata.setdefault(key, getattr(corpus, key))
                if metadata[key] is None:
                    metadata[key] = getattr(corpus, key)
        return store.create_dataset(
            name=name,
            kind="local",
            corpus_id=corpus_id,
            status="ready",
            progress=1.0,
            path=str(directory.resolve()),
            manifest=manifest,
            **{key: value for key, value in metadata.items() if value is not None},
            **_summary_fields(manifest),
        )

    def upload(self, name: str, files: list[tuple[str, bytes]]) -> Dataset:
        if not files:
            raise ValueError("At least one audio file is required.")
        for filename, _ in files:
            if Path(filename).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
                raise ValueError(
                    f"Unsupported file type for {filename!r}; allowed: {sorted(ALLOWED_UPLOAD_SUFFIXES)}"
                )
        dataset_id = uuid.uuid4()
        directory = datasets_dir() / str(dataset_id)
        directory.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        for filename, payload in files:
            safe = _safe_filename(filename, used)
            used.add(safe)
            (directory / safe).write_bytes(payload)
        manifest = scan_directory(directory)
        if not manifest["files"]:
            shutil.rmtree(directory, ignore_errors=True)
            raise ValueError("None of the uploaded files could be read as audio.")
        return store.create_dataset(
            id=dataset_id,
            name=name,
            kind="upload",
            status="ready",
            progress=1.0,
            path=str(directory),
            manifest=manifest,
            **_summary_fields(manifest),
        )

    def synthetic(self, name: str | None, sample_rate: int, duration_seconds: float, seed: int) -> Dataset:
        dataset_id = uuid.uuid4()
        directory = datasets_dir() / str(dataset_id)
        manifest = write_synthetic_set(directory, sample_rate, duration_seconds, seed)
        return store.create_dataset(
            id=dataset_id,
            name=name or f"Synthetic test signals ({sample_rate / 1000:g} kHz, seed {seed})",
            kind="synthetic",
            status="ready",
            progress=1.0,
            path=str(directory),
            domain="synthetic_signal",
            license="Generated (public domain)",
            description="Tones, a log sweep, white and pink noise, modulated noise, tone bursts in silence, a full-scale square wave and low-level noise.",
            rule={"sample_rate": sample_rate, "duration_seconds": duration_seconds, "seed": seed},
            manifest=manifest,
            **_summary_fields(manifest),
        )

    def delete(self, dataset_id: uuid.UUID) -> bool:
        task = self._tasks.pop(dataset_id, None)
        if task is not None and not task.done():
            task.cancel()
        dataset = store.delete_dataset(dataset_id)
        if dataset is None:
            return False
        if dataset.kind in OWNED_KINDS and dataset.path:
            shutil.rmtree(dataset.path, ignore_errors=True)
        return True


def _safe_filename(filename: str, used: set[str]) -> str:
    original = Path(filename)
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in original.stem).strip("._") or "audio"
    candidate = f"{stem}{original.suffix.lower()}"
    counter = 1
    while candidate in used:
        candidate = f"{stem}_{counter}{original.suffix.lower()}"
        counter += 1
    return candidate


library = Library()

__all__ = ["Library", "library", "resolve"]
