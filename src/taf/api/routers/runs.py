"""Runs: progress, results, statistics, exports and trial inspection."""

from __future__ import annotations

import asyncio
import json
import uuid
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from taf.experiments.csv_export import export_detailed_csv, export_summary_csv, make_export_filename
from taf.experiments.results import ExperimentResultRow
from taf.experiments.schema import ExperimentConfig
from taf.persistence import store
from taf.persistence.models import Experiment, Run

from ..runs import runs
from ..schemas import RowRecord, RowsPage, RunOut

router = APIRouter(prefix="/api/runs", tags=["runs"])

_SSE_HEARTBEAT_SECONDS = 15.0
INSPECT_SIGNALS = ("cover", "stego", "attacked", "residual")


def run_out(run: Run, experiment: Experiment) -> RunOut:
    return RunOut(
        id=run.id,
        number=run.number,
        status=run.status,
        experiment_version=run.experiment_version,
        total_rows=run.total_rows,
        completed_rows=run.completed_rows,
        ok_rows=run.ok_rows,
        error=run.error,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        experiment_id=experiment.id,
        experiment_name=experiment.name,
        experiment_type=experiment.experiment_type,
        config=run.config,
        plan=run.plan or {},
    )


def _require(run_id: uuid.UUID) -> tuple[Run, Experiment]:
    run = store.get_run(run_id)
    experiment = store.get_experiment(run.experiment_id) if run else None
    if run is None or experiment is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run, experiment


def _config(run: Run, experiment: Experiment) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {**run.config, "name": experiment.name, "experiment_type": experiment.experiment_type}
    )


def _download(content: str | bytes, filename: str, media_type: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _sse(event_type: str, data: Any) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


# --------------------------------------------------------------------------
# Listing and progress
# --------------------------------------------------------------------------


@router.get("", response_model=list[RunOut])
def list_runs(limit: int = Query(100, ge=1, le=1000)) -> list[RunOut]:
    return [run_out(run, experiment) for run, experiment in store.list_runs(limit=limit)]


@router.get("/events")
async def stream_all(max_events: int | None = Query(None, ge=0)) -> StreamingResponse:
    """Status changes of every run, for dashboards."""

    async def events():
        queue = runs.subscribe_all()
        sent = 0
        try:
            yield _sse("hello", {})
            while max_events is None or sent < max_events:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield _sse(event["type"], event)
                sent += 1
        finally:
            runs.unsubscribe_all(queue)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: uuid.UUID) -> RunOut:
    return run_out(*_require(run_id))


@router.get("/{run_id}/summary")
def get_summary(run_id: uuid.UUID) -> JSONResponse:
    run, _ = _require(run_id)
    return JSONResponse({"run_id": str(run.id), "status": run.status, "summary": run.summary or {}})


@router.get("/{run_id}/manifest.json")
def get_manifest(run_id: uuid.UUID, download: bool = Query(True)) -> Response:
    run, experiment = _require(run_id)
    if not run.manifest:
        raise HTTPException(status_code=409, detail="The run has not produced a manifest yet.")
    body = json.dumps(run.manifest, indent=2, default=str)
    if not download:
        return Response(content=body, media_type="application/json")
    return _download(body, f"{experiment.experiment_type}_{run.id}_manifest.json", "application/json")


@router.get("/{run_id}/config.json")
def get_config(run_id: uuid.UUID) -> Response:
    run, experiment = _require(run_id)
    body = json.dumps(_config(run, experiment).model_dump(mode="json"), indent=2, default=str)
    return _download(body, f"{experiment.experiment_type}_{run.id}_config.json", "application/json")


@router.get("/{run_id}/rows", response_model=RowsPage)
def get_rows(
    run_id: uuid.UUID,
    method: str | None = None,
    attack: str | None = Query(None, description="Attack specification; empty string for the no-attack baseline."),
    status_filter: str | None = Query(None, alias="status"),
    file_name: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> RowsPage:
    _require(run_id)
    total, records = store.query_rows(
        run_id, method=method, attack=attack, status=status_filter, file_name=file_name, offset=offset, limit=limit
    )
    return RowsPage(
        total=total,
        offset=offset,
        limit=limit,
        rows=[RowRecord(id=row_id, row=ExperimentResultRow.model_validate(data)) for row_id, data in records],
    )


@router.get("/{run_id}/facets")
def get_facets(run_id: uuid.UUID) -> dict[str, list[Any]]:
    _require(run_id)
    return store.row_facets(run_id)


@router.get("/{run_id}/events")
async def stream_run(run_id: uuid.UUID) -> StreamingResponse:
    """Snapshot of the run, then its rows and status changes as they happen."""
    run, experiment = _require(run_id)

    async def events():
        queue = runs.subscribe(run_id)
        try:
            yield _sse("snapshot", run_out(*_require(run_id)).model_dump(mode="json"))
            if not runs.is_active(run_id):
                yield _sse("done", {})
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield _sse(event["type"], event)
                if event["type"] == "done":
                    return
        finally:
            runs.unsubscribe(run_id, queue)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.post("/{run_id}/cancel", response_model=RunOut)
def cancel_run(run_id: uuid.UUID) -> RunOut:
    _require(run_id)
    if not runs.cancel(run_id):
        raise HTTPException(status_code=409, detail="The run is not active.")
    return run_out(*_require(run_id))


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(run_id: uuid.UUID) -> None:
    if runs.is_active(run_id):
        raise HTTPException(status_code=409, detail="Cancel the run before deleting it.")
    if not store.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


# --------------------------------------------------------------------------
# Exports and reports
# --------------------------------------------------------------------------


@router.get("/{run_id}/export.csv")
def export_rows(run_id: uuid.UUID) -> Response:
    run, experiment = _require(run_id)
    filename = make_export_filename(experiment.experiment_type, str(run.id)[:8], "detailed", run.created_at)
    return _download(export_detailed_csv(store.all_rows(run_id)), filename, "text/csv")


@router.get("/{run_id}/export_summary.csv")
def export_summary(run_id: uuid.UUID) -> Response:
    run, experiment = _require(run_id)
    filename = make_export_filename(experiment.experiment_type, str(run.id)[:8], "summary", run.created_at)
    return _download(export_summary_csv(run.summary or {}), filename, "text/csv")


@router.get("/{run_id}/report.{extension}")
def export_report(run_id: uuid.UUID, extension: str, download: bool = Query(True)) -> Response:
    """Experimental-setup paragraph and result tables, as LaTeX or Markdown."""
    from taf.experiments.reporting import build_report

    if extension not in ("tex", "md"):
        raise HTTPException(status_code=404, detail="Reports are available as .tex and .md")
    run, experiment = _require(run_id)
    if run.status != "completed":
        raise HTTPException(status_code=409, detail="Reports are available for completed runs.")
    text = build_report(
        f"{experiment.name} — run {run.number}",
        _config(run, experiment),
        run.manifest or {},
        run.summary or {},
        "latex" if extension == "tex" else "markdown",
    )
    media = "application/x-tex" if extension == "tex" else "text/markdown"
    if not download:
        return PlainTextResponse(text, media_type=media)
    return _download(text, f"{experiment.experiment_type}_run{run.number}_report.{extension}", media)


# --------------------------------------------------------------------------
# Trial inspection
# --------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _trial(run_id: uuid.UUID, row_id: int):
    from taf.experiments.inspector import resynthesize

    run, experiment = _require(run_id)
    data = store.get_row(run_id, row_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Row not found: {row_id}")
    row = ExperimentResultRow.model_validate(data)
    return row, resynthesize(_config(run, experiment), row)


def _trial_or_error(run_id: uuid.UUID, row_id: int):
    try:
        return _trial(run_id, row_id)
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/{run_id}/rows/{row_id}/inspect")
async def inspect_row(run_id: uuid.UUID, row_id: int) -> JSONResponse:
    """Re-synthesised signals of one trial: spectrograms, waveforms, reproduction check."""
    from taf.experiments.inspector import envelope, peak_magnitude, spectrogram

    row, trial = await asyncio.to_thread(_trial_or_error, run_id, row_id)
    reference = peak_magnitude(trial.cover, trial.sample_rate)
    signals = {"cover": trial.cover, "stego": trial.stego, "residual": trial.residual}
    if trial.attacked is not None:
        signals["attacked"] = trial.attacked
    payload = {
        "row": row.model_dump(mode="json"),
        "sample_rate": trial.sample_rate,
        "duration_seconds": len(trial.cover) / trial.sample_rate,
        "reproduced": trial.reproduced,
        "decoded_bits": trial.decoded_bits,
        "attack_metadata": trial.attack_metadata,
        "signals": {
            name: {
                "spectrogram": spectrogram(signal, trial.sample_rate, reference),
                "envelope": envelope(signal),
            }
            for name, signal in signals.items()
        },
    }
    return JSONResponse(store.json_safe(payload))


@router.get("/{run_id}/rows/{row_id}/audio/{signal}.wav")
async def row_audio(run_id: uuid.UUID, row_id: int, signal: str) -> Response:
    """One signal of a trial as 16-bit WAV. The residual is normalised to -1 dBFS."""
    from taf.experiments.inspector import wav_bytes

    if signal not in INSPECT_SIGNALS:
        raise HTTPException(status_code=404, detail=f"Unknown signal {signal!r}; one of {INSPECT_SIGNALS}")
    _, trial = await asyncio.to_thread(_trial_or_error, run_id, row_id)
    data = {"cover": trial.cover, "stego": trial.stego, "attacked": trial.attacked, "residual": trial.residual}[signal]
    if data is None:
        raise HTTPException(status_code=404, detail="This trial has no attack.")
    body, gain_db = wav_bytes(data, trial.sample_rate, -1.0 if signal == "residual" else None)
    return Response(content=body, media_type="audio/wav", headers={"X-Gain-dB": f"{gain_db:.1f}"})
