"""Engine, sessions and schema migrations."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from taf.persistence.settings import database_url

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


@lru_cache(maxsize=None)
def engine(url: str | None = None) -> Engine:
    return create_engine(url or database_url(), pool_pre_ping=True, future=True)


@lru_cache(maxsize=None)
def _session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine(url), expire_on_commit=False)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """A transaction: committed on success, rolled back on error."""
    session = _session_factory(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def alembic_config(url: str | None = None):
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", (url or database_url()).replace("%", "%%"))
    return config


def upgrade(url: str | None = None) -> None:
    """Bring the schema to the latest migration."""
    from alembic import command

    command.upgrade(alembic_config(url), "head")


def check_connection(url: str | None = None) -> str:
    """Server version string; raises when the database is unreachable."""
    with engine(url).connect() as connection:
        return str(connection.execute(text("select version()")).scalar())


def reset_caches() -> None:
    """Forget engines, e.g. after changing ``TAF_DATABASE_URL`` in tests."""
    _session_factory.cache_clear()
    engine.cache_clear()


__all__ = ["MIGRATIONS_DIR", "alembic_config", "check_connection", "engine", "reset_caches", "session_scope", "upgrade"]
