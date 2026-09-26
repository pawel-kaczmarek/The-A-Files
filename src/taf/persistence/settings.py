"""Where the platform keeps its database and its files."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATABASE_URL = "postgresql+psycopg://taf:taf@localhost:5432/taf"


def database_url() -> str:
    """``TAF_DATABASE_URL``, defaulting to the PostgreSQL of ``docker-compose.yml``."""
    return os.environ.get("TAF_DATABASE_URL", DEFAULT_DATABASE_URL)


def data_dir() -> Path:
    """Root for downloaded archives, prepared corpora and uploads (``TAF_DATA_DIR``)."""
    root = Path(os.environ.get("TAF_DATA_DIR", Path.home() / ".taf")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def datasets_dir() -> Path:
    path = data_dir() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def downloads_dir() -> Path:
    path = data_dir() / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["DEFAULT_DATABASE_URL", "data_dir", "database_url", "datasets_dir", "downloads_dir"]
