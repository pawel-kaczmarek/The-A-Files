from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if __package__ in (None, ""):
    # Executed directly (python src/taf/api/app.py): import taf from src/,
    # not from a possibly stale site-packages install.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from taf import __version__
from taf.api.routers import catalog, datasets, experiments

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def create_app() -> FastAPI:
    app = FastAPI(
        title="The A-Files API",
        version=__version__,
        description=(
            "REST interface over the The A-Files audio steganography evaluation engine: "
            "catalog discovery, experiment execution and live result streaming."
        ),
    )

    origins = os.environ.get("TAF_API_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(catalog.router)
    app.include_router(datasets.router)
    app.include_router(experiments.router)

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("TAF_API_HOST", "127.0.0.1")
    port = int(os.environ.get("TAF_API_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
