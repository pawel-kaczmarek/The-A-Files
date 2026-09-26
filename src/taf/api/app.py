from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__ in (None, ""):
    # Executed directly (python src/taf/api/app.py): import taf from src/,
    # not from a possibly stale site-packages install.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loguru import logger

from taf import __version__
from taf.api.routers import catalog, datasets, experiments, runs as run_routes
from taf.api.schemas import PlatformStats, SystemStatus

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from taf.api import library  # noqa: F401 - registers the library dataset resolver
    from taf.api.runs import runs
    from taf.persistence import session, store

    try:
        session.upgrade()
    except Exception as error:  # noqa: BLE001 - re-raised with guidance
        from taf.persistence.settings import database_url

        raise RuntimeError(
            f"The platform database is not reachable at {database_url()}. Start PostgreSQL "
            "(docker compose up -d db) or set TAF_DATABASE_URL."
        ) from error
    interrupted = store.mark_interrupted_runs()
    stale = store.mark_interrupted_datasets()
    if interrupted or stale:
        logger.warning("Marked {} run(s) and {} dataset(s) as interrupted", interrupted, stale)
    yield
    await runs.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="The A-Files Research Platform API",
        version=__version__,
        description=(
            "Design, run and analyse experiments on audio steganography and watermarking "
            "methods: imperceptibility, robustness, capacity and detectability."
        ),
        lifespan=lifespan,
    )

    origins = os.environ.get("TAF_API_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "X-Gain-dB"],
    )

    app.include_router(catalog.router)
    app.include_router(datasets.router)
    app.include_router(experiments.router)
    app.include_router(run_routes.router)

    @app.get("/api/health", response_model=SystemStatus, tags=["system"])
    def health() -> SystemStatus:
        from taf.persistence.session import check_connection
        from taf.persistence.settings import data_dir

        try:
            database = {"ok": True, "version": check_connection().split(" on ")[0]}
        except Exception as error:  # noqa: BLE001 - reported, not raised
            database = {"ok": False, "error": str(error).splitlines()[0]}
        return SystemStatus(
            status="ok" if database["ok"] else "degraded",
            version=__version__,
            database=database,
            data_dir=str(data_dir()),
        )

    @app.get("/api/stats", response_model=PlatformStats, tags=["system"])
    def stats() -> PlatformStats:
        from taf.experiments.registry import list_designs
        from taf.persistence import store

        return PlatformStats(
            experiments=len(store.list_experiments(include_archived=True)),
            runs=store.run_counts(),
            datasets=len(store.list_datasets()),
            methods=len(catalog.list_methods()),
            metrics=len(catalog.list_metrics()),
            attacks=len(catalog.list_attacks()),
            designs=len(list_designs()),
        )

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("TAF_API_HOST", "127.0.0.1")
    port = int(os.environ.get("TAF_API_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
