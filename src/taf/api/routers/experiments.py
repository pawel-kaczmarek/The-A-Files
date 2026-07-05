from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from taf.experiments.csv_export import export_detailed_csv, export_summary_csv, make_export_filename
from taf.experiments.runner import preview_experiment, validate_config
from taf.experiments.schema import UPLOAD_DATASET_PREFIX

from ..jobs import Job, registry
from ..schemas import (
    ExperimentConfig,
    ExperimentDetail,
    ExperimentPlan,
    ExperimentResultRow,
    JobSummary,
)
from ..uploads import upload_registry

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

_SSE_HEARTBEAT_SECONDS = 15.0


@router.post("/preview", response_model=ExperimentPlan)
def preview(config: ExperimentConfig) -> ExperimentPlan:
    return preview_experiment(config)


@router.post("/run", response_model=JobSummary, status_code=status.HTTP_201_CREATED)
async def run_experiment(config: ExperimentConfig) -> JobSummary:
    if config.dataset_id and config.dataset_id.startswith(UPLOAD_DATASET_PREFIX):
        upload_id = config.dataset_id[len(UPLOAD_DATASET_PREFIX):]
        if upload_registry.get(upload_id) is None:
            raise HTTPException(status_code=422, detail=f"Uploaded dataset not found: {upload_id}")
    problems = validate_config(config)
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems))
    job = registry.create(config)
    return job.job_summary()


@router.get("/history", response_model=list[JobSummary])
def history() -> list[JobSummary]:
    return [job.job_summary() for job in registry.list()]


# Alias kept so a bare GET /api/experiments also lists runs.
@router.get("", response_model=list[JobSummary])
def list_experiments() -> list[JobSummary]:
    return history()


# Registered before /{experiment_id} so the path segment "events" is not
# captured as an experiment id.
@router.get("/events")
async def stream_all_experiments(max_events: int | None = None) -> StreamingResponse:
    """Global SSE stream: a snapshot of all experiments, then live summary updates.

    ``max_events`` ends the stream after that many update events (0 = snapshot
    only) — handy for curl debugging and deterministic tests.
    """

    async def event_stream():
        queue = registry.subscribe()
        sent = 0
        try:
            snapshot = [job.job_summary().model_dump(mode="json") for job in registry.list()]
            yield _sse_event("experiments", snapshot)
            while max_events is None or sent < max_events:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield _sse_event(event["type"], event["summary"])
                sent += 1
        finally:
            registry.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{experiment_id}", response_model=JobSummary)
def get_experiment(experiment_id: str) -> JobSummary:
    return _require_job(experiment_id).job_summary()


@router.get("/{experiment_id}/results", response_model=list[ExperimentResultRow])
def get_results(experiment_id: str) -> list[ExperimentResultRow]:
    return _require_job(experiment_id).rows


@router.get("/{experiment_id}/detail", response_model=ExperimentDetail)
def get_detail(experiment_id: str) -> ExperimentDetail:
    return _require_job(experiment_id).detail()


@router.get("/{experiment_id}/summary")
def get_summary(experiment_id: str) -> JSONResponse:
    job = _require_job(experiment_id)
    return JSONResponse(
        {"experiment_id": job.id, "status": job.status, "summary": _json_safe(job.summary)}
    )


@router.get("/{experiment_id}/config.json")
def export_config(experiment_id: str) -> JSONResponse:
    job = _require_job(experiment_id)
    return JSONResponse(
        job.config.model_dump(mode="json"),
        headers={
            "Content-Disposition": f'attachment; filename="{job.config.experiment_type.value}_{job.id}_config.json"'
        },
    )


@router.get("/{experiment_id}/export.csv", response_class=PlainTextResponse)
def export_csv(experiment_id: str) -> PlainTextResponse:
    job = _require_job(experiment_id)
    filename = make_export_filename(job.config.experiment_type.value, job.id, "detailed", job.created_at)
    return PlainTextResponse(
        export_detailed_csv(job.rows),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{experiment_id}/export_summary.csv", response_class=PlainTextResponse)
def export_summary(experiment_id: str) -> PlainTextResponse:
    job = _require_job(experiment_id)
    filename = make_export_filename(job.config.experiment_type.value, job.id, "summary", job.created_at)
    return PlainTextResponse(
        export_summary_csv(_json_safe(job.summary)),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{experiment_id}/events")
async def stream_events(experiment_id: str) -> StreamingResponse:
    job = _require_job(experiment_id)

    async def event_stream():
        queue = job.subscribe()
        try:
            yield _sse_event("snapshot", job.detail().model_dump(mode="json"))
            if job.status in {"completed", "failed"}:
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield _sse_event(event["type"], event.get("summary") or event.get("row") or {})
                if event["type"] == "done":
                    return
        finally:
            job.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_event(event_type: str, data: dict | list) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _json_safe(value):
    """Replace non-finite floats with None so JSONResponse never emits NaN."""
    import math

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _require_job(experiment_id: str) -> Job:
    job = registry.get(experiment_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return job
