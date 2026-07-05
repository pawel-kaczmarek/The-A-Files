"""In-process registry of experiment runs with live event broadcasting."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from taf.experiments.results import ExperimentResultRow
from taf.experiments.runner import preview_experiment, run_experiment_async
from taf.experiments.schema import ExperimentConfig

from .schemas import ExperimentDetail, JobSummary


@dataclass
class Job:
    id: str
    config: ExperimentConfig
    created_at: datetime
    status: str = "pending"
    total_tasks: int = 0
    rows: list[ExperimentResultRow] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    task: asyncio.Task | None = None
    last_global_emit: float = 0.0

    def job_summary(self) -> JobSummary:
        return JobSummary(
            experiment_id=self.id,
            experiment_type=self.config.experiment_type,
            name=self.config.name,
            status=self.status,  # type: ignore[arg-type]
            created_at=self.created_at,
            dataset=self.config.dataset_id or self.config.dataset_path,
            methods=self.config.methods,
            metrics=self.config.metrics,
            attacks=self.config.attacks,
            payload_lengths=self.config.payload_lengths,
            repetitions=self.config.repetitions,
            total_tasks=self.total_tasks,
            completed_rows=len(self.rows),
            success_rows=sum(1 for row in self.rows if row.decode_success),
            error=self.error,
            csv_url=f"/api/experiments/{self.id}/export.csv",
            summary_csv_url=f"/api/experiments/{self.id}/export_summary.csv",
            config=self.config,
        )

    def detail(self) -> ExperimentDetail:
        return ExperimentDetail(
            **self.job_summary().model_dump(),
            rows=self.rows,
            summary=self.summary,
        )

    def broadcast(self, event: dict[str, Any]) -> None:
        for queue in list(self.subscribers):
            queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)


class JobRegistry:
    """Holds all experiment jobs plus registry-wide subscribers for the global stream."""

    _GLOBAL_EMIT_INTERVAL_SECONDS = 0.5

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._subscribers: list[asyncio.Queue] = []

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _emit_global(self, job: Job, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - job.last_global_emit < self._GLOBAL_EMIT_INTERVAL_SECONDS:
            return
        job.last_global_emit = now
        event = {"type": "experiment", "summary": job.job_summary().model_dump(mode="json")}
        for queue in list(self._subscribers):
            queue.put_nowait(event)

    def create(self, config: ExperimentConfig) -> Job:
        job_id = config.experiment_id or uuid.uuid4().hex[:12]
        config = config.model_copy(update={"experiment_id": job_id})
        job = Job(id=job_id, config=config, created_at=datetime.now(timezone.utc))
        self._jobs[job_id] = job
        self._emit_global(job, force=True)
        job.task = asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: Job) -> None:
        job.status = "running"
        job.total_tasks = preview_experiment(job.config).estimated_result_rows
        job.broadcast({"type": "status", "summary": job.job_summary().model_dump(mode="json")})
        self._emit_global(job, force=True)

        def on_row(row: ExperimentResultRow) -> None:
            job.rows.append(row)
            job.broadcast({"type": "row", "row": row.model_dump(mode="json")})
            self._emit_global(job)

        try:
            run = await run_experiment_async(job.config, on_row=on_row)
            # The runner appends rows itself; ours came through on_row already.
            job.summary = run.summary
            job.status = run.status
            job.error = run.error
        except Exception as error:  # pragma: no cover - defensive path
            job.status = "failed"
            job.error = str(error)
        finally:
            job.broadcast({"type": "done", "summary": job.job_summary().model_dump(mode="json")})
            self._emit_global(job, force=True)


registry = JobRegistry()
