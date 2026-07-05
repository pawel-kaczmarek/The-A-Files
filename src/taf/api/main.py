"""Uvicorn entry point: ``uvicorn taf.api.main:app --reload``."""

from taf.api.app import create_app

app = create_app()
