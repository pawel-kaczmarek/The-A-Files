"""Execution of experiment runs with persistence and live event streams.

A run is executed in the API process by the experiment engine. Its rows are
written to the database in small batches while it runs, so a long run can
be inspected before it finishes and nothing is lost if the browser closes;
subscribers receive every row and status change as it happens. Runs that
were active when the server stopped are marked ``interrupted`` at start-up.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from taf.experiments.results import ExperimentResultRow
from taf.experiments.runner import preview_experiment, run_experiment_async, validate_config
from taf.experiments.schema import ExperimentConfig
from taf.persistence import store

FLUSH_INTERVAL_SECONDS = 0.5


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    def __init__(self) -> None:
        self._tasks: dict[uuid.UUID, asyncio.Task] = {}
        self._subscribers: dict[uuid.UUID, list[asyncio.Queue]] = {}
        self._global: list[asyncio.Queue] = []
        self._semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------ events

    def subscribe(self, run_id: uuid.UUID) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, []).append(queue)
        return queue

    def unsubscribe(self, run_id: uuid.UUID, queue: asyncio.Queue) -> None:
        queues = self._subscribers.get(run_id, [])
        if queue in queues:
            queues.remove(queue)

    def subscribe_all(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._global.append(queue)
        return queue

    def unsubscribe_all(self, queue: asyncio.Queue) -> None:
        if queue in self._global:
            self._global.remove(queue)

    def _emit(self, run_id: uuid.UUID, event: dict[str, Any], broadcast: bool = False) -> None:
        for queue in list(self._subscribers.get(run_id, [])):
            queue.put_nowait(event)
        if broadcast:
            for queue in list(self._global):
                queue.put_nowait({**event, "run_id": str(run_id)})

    def is_active(self, run_id: uuid.UUID) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    # ------------------------------------------------------------ control

    def start(self, experiment_id: uuid.UUID) -> uuid.UUID:
        """Queue a run of the experiment's current protocol; raises ``ValueError``."""
        experiment = store.get_experiment(experiment_id)
        if experiment is None:
            raise LookupError(f"Experiment not found: {experiment_id}")
        config = ExperimentConfig.model_validate(
            {**experiment.config, "name": experiment.name, "experiment_type": experiment.experiment_type}
        )
        problems = validate_config(config)
        if problems:
            raise ValueError("; ".join(problems))
        run = store.create_run(experiment_id)
        assert run is not None
        self._tasks[run.id] = asyncio.create_task(self._execute(run.id, config))
        self._emit(run.id, {"type": "status", "status": "queued"}, broadcast=True)
        return run.id

    def cancel(self, run_id: uuid.UUID) -> bool:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _execute(self, run_id: uuid.UUID, config: ExperimentConfig) -> None:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(int(os.environ.get("TAF_MAX_CONCURRENT_RUNS", "1")))
        buffer: list[ExperimentResultRow] = []
        written = 0

        async def flush() -> None:
            nonlocal written
            if not buffer:
                return
            batch = buffer[:]
            buffer.clear()
            await asyncio.to_thread(store.append_rows, run_id, batch, written)
            written += len(batch)

        async def flusher() -> None:
            while True:
                await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
                await flush()

        def on_row(row: ExperimentResultRow) -> None:
            buffer.append(row)
            self._emit(run_id, {"type": "row", "row": store.json_safe(row.model_dump(mode="json"))})

        flushing: asyncio.Task | None = None
        try:
            async with self._semaphore:
                plan = await asyncio.to_thread(preview_experiment, config)
                config = config.model_copy(update={"experiment_id": str(run_id)})
                await asyncio.to_thread(
                    store.update_run,
                    run_id,
                    status="running",
                    started_at=_now(),
                    total_rows=plan.estimated_result_rows,
                    plan=plan.model_dump(mode="json"),
                )
                self._emit(run_id, {"type": "status", "status": "running", "total_rows": plan.estimated_result_rows}, broadcast=True)

                flushing = asyncio.create_task(flusher())
                result = await run_experiment_async(config, on_row=on_row)
                flushing.cancel()
                await flush()
                await asyncio.to_thread(
                    store.update_run,
                    run_id,
                    status=result.status,
                    error=result.error,
                    summary=result.summary,
                    manifest=result.manifest,
                    # The resolved configuration carries the drawn seed.
                    config=result.config.model_dump(mode="json", exclude={"created_at"}),
                    finished_at=_now(),
                )
                self._emit(run_id, {"type": "status", "status": result.status, "error": result.error}, broadcast=True)
        except asyncio.CancelledError:
            if flushing is not None:
                flushing.cancel()
            await flush()
            await asyncio.to_thread(store.update_run, run_id, status="cancelled", finished_at=_now())
            self._emit(run_id, {"type": "status", "status": "cancelled"}, broadcast=True)
        except Exception as error:  # noqa: BLE001 - recorded on the run
            logger.exception("Run {} failed: {}", run_id, error)
            if flushing is not None:
                flushing.cancel()
            await asyncio.to_thread(store.update_run, run_id, status="failed", error=str(error), finished_at=_now())
            self._emit(run_id, {"type": "status", "status": "failed", "error": str(error)}, broadcast=True)
        finally:
            self._emit(run_id, {"type": "done"})
            self._tasks.pop(run_id, None)

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            try:
                await task
            except BaseException:  # noqa: BLE001 - shutting down
                pass


runs = RunManager()

__all__ = ["RunManager", "runs"]
