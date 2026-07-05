"""In-process registry for user-uploaded audio datasets.

Uploaded files are stored under the system temp directory and referenced from
experiments through the ``upload:<id>`` dataset name.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from taf.audio.formats import AudioFileFormat

UPLOAD_DATASET_PREFIX = "upload:"

_ALLOWED_EXTENSIONS = {
    AudioFileFormat.WAV.extension,
    AudioFileFormat.FLAC.extension,
    AudioFileFormat.OGG.extension,
}


@dataclass
class UploadedDataset:
    id: str
    name: str
    directory: Path
    created_at: datetime
    files: list[Path] = field(default_factory=list)

    @property
    def dataset_name(self) -> str:
        return f"{UPLOAD_DATASET_PREFIX}{self.id}"


class UploadRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path(tempfile.gettempdir()) / "taf-api" / "uploads"
        self._datasets: dict[str, UploadedDataset] = {}

    def list(self) -> list[UploadedDataset]:
        return sorted(self._datasets.values(), key=lambda item: item.created_at, reverse=True)

    def get(self, upload_id: str) -> UploadedDataset | None:
        return self._datasets.get(upload_id)

    def create(self, name: str | None, files: list[tuple[str, bytes]]) -> UploadedDataset:
        if not files:
            raise ValueError("At least one audio file is required.")
        for filename, _ in files:
            suffix = Path(filename).suffix.lower()
            if suffix not in _ALLOWED_EXTENSIONS:
                allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
                raise ValueError(f"Unsupported file type '{suffix}' for '{filename}'. Allowed: {allowed}")

        upload_id = uuid.uuid4().hex[:12]
        directory = self._root / upload_id
        directory.mkdir(parents=True, exist_ok=True)

        stored: list[Path] = []
        used_names: set[str] = set()
        for filename, payload in files:
            safe_name = _safe_filename(filename, used_names)
            used_names.add(safe_name)
            path = directory / safe_name
            path.write_bytes(payload)
            stored.append(path)

        dataset = UploadedDataset(
            id=upload_id,
            name=(name or "").strip() or f"upload-{upload_id}",
            directory=directory,
            created_at=datetime.now(timezone.utc),
            files=sorted(stored),
        )
        self._datasets[upload_id] = dataset
        return dataset

    def delete(self, upload_id: str) -> bool:
        dataset = self._datasets.pop(upload_id, None)
        if dataset is None:
            return False
        shutil.rmtree(dataset.directory, ignore_errors=True)
        return True


def _safe_filename(filename: str, used_names: set[str]) -> str:
    original = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", original.stem).strip("._") or "audio"
    suffix = original.suffix.lower()
    candidate = f"{stem}{suffix}"
    counter = 1
    while candidate in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


upload_registry = UploadRegistry()
