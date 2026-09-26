"""Data access for experiments, runs, result rows and datasets.

Functions take and return ORM objects detached from their session
(``expire_on_commit=False``), so callers can read them after the transaction
ends. All writes go through here, which keeps the versioning and numbering
rules in one place.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import delete, func, select, update

from taf.experiments.results import ExperimentResultRow
from taf.persistence.models import Dataset, Experiment, ResultRowRecord, Run
from taf.persistence.session import session_scope

ACTIVE_RUN_STATES = ("queued", "running")
FINAL_RUN_STATES = ("completed", "failed", "cancelled", "interrupted")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def json_safe(value: Any) -> Any:
    """``value`` with non-finite floats replaced by ``None``.

    JSONB has no NaN or infinity, and summaries and attack metadata can
    contain both (an infinite measured SNR, an undefined statistic).
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        return json_safe(value.item())
    return value


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def create_experiment(**fields: Any) -> Experiment:
    with session_scope() as session:
        experiment = Experiment(**fields)
        session.add(experiment)
        session.flush()
        return experiment


def get_experiment(experiment_id: uuid.UUID) -> Experiment | None:
    with session_scope() as session:
        return session.get(Experiment, experiment_id)


def list_experiments(include_archived: bool = False) -> list[tuple[Experiment, Run | None, int]]:
    """Experiments, newest first, each with its latest run and run count."""
    with session_scope() as session:
        query = select(Experiment).order_by(Experiment.updated_at.desc())
        if not include_archived:
            query = query.where(Experiment.archived.is_(False))
        experiments = list(session.scalars(query))
        if not experiments:
            return []
        ids = [experiment.id for experiment in experiments]
        counts = dict(
            session.execute(
                select(Run.experiment_id, func.count()).where(Run.experiment_id.in_(ids)).group_by(Run.experiment_id)
            ).all()
        )
        latest: dict[uuid.UUID, Run] = {}
        for run in session.scalars(select(Run).where(Run.experiment_id.in_(ids)).order_by(Run.created_at)):
            latest[run.experiment_id] = run
        return [(experiment, latest.get(experiment.id), counts.get(experiment.id, 0)) for experiment in experiments]


def update_experiment(experiment_id: uuid.UUID, **fields: Any) -> Experiment | None:
    """Apply ``fields``; a changed configuration or design starts a new version."""
    with session_scope() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            return None
        changes_protocol = (
            "config" in fields and fields["config"] != experiment.config
        ) or (
            "experiment_type" in fields and fields["experiment_type"] != experiment.experiment_type
        )
        for key, value in fields.items():
            setattr(experiment, key, value)
        if changes_protocol:
            experiment.version += 1
        experiment.updated_at = _now()
        session.flush()
        return experiment


def delete_experiment(experiment_id: uuid.UUID) -> bool:
    with session_scope() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            return False
        session.delete(experiment)
        return True


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def create_run(experiment_id: uuid.UUID) -> Run | None:
    """A queued run of the experiment's current protocol version."""
    with session_scope() as session:
        experiment = session.get(Experiment, experiment_id)
        if experiment is None:
            return None
        number = (
            session.scalar(select(func.max(Run.number)).where(Run.experiment_id == experiment_id)) or 0
        ) + 1
        run = Run(
            experiment_id=experiment_id,
            number=number,
            experiment_version=experiment.version,
            status="queued",
            config=dict(experiment.config),
        )
        session.add(run)
        experiment.updated_at = _now()
        session.flush()
        return run


def get_run(run_id: uuid.UUID) -> Run | None:
    with session_scope() as session:
        return session.get(Run, run_id)


def list_runs(experiment_id: uuid.UUID | None = None, limit: int = 200) -> list[tuple[Run, Experiment]]:
    with session_scope() as session:
        query = select(Run, Experiment).join(Experiment).order_by(Run.created_at.desc()).limit(limit)
        if experiment_id is not None:
            query = query.where(Run.experiment_id == experiment_id)
        return [(run, experiment) for run, experiment in session.execute(query).all()]


def update_run(run_id: uuid.UUID, **fields: Any) -> None:
    with session_scope() as session:
        session.execute(
            update(Run).where(Run.id == run_id).values(**{key: json_safe(value) for key, value in fields.items()})
        )


def append_rows(run_id: uuid.UUID, rows: Sequence[ExperimentResultRow], first_ordinal: int) -> None:
    """Store trial rows and advance the run's counters in one transaction."""
    if not rows:
        return
    with session_scope() as session:
        session.add_all(
            ResultRowRecord(
                run_id=run_id,
                ordinal=first_ordinal + offset,
                file_name=row.file_name,
                method=row.method,
                method_name=row.method_type,
                payload_length=row.payload_length,
                repetition=row.repetition,
                attack=row.attack,
                status=row.status,
                failure_kind=row.failure_kind,
                ber=row.ber,
                decode_success=row.decode_success,
                data=json_safe(row.model_dump(mode="json")),
            )
            for offset, row in enumerate(rows)
        )
        session.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(
                completed_rows=Run.completed_rows + len(rows),
                ok_rows=Run.ok_rows + sum(1 for row in rows if row.status == "ok"),
            )
        )


def query_rows(
    run_id: uuid.UUID,
    *,
    method: str | None = None,
    attack: str | None = None,
    status: str | None = None,
    file_name: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> tuple[int, list[tuple[int, dict[str, Any]]]]:
    """``(total matching, [(row id, row data)])`` in trial order."""
    with session_scope() as session:
        query = select(ResultRowRecord).where(ResultRowRecord.run_id == run_id)
        if method is not None:
            query = query.where(ResultRowRecord.method == method)
        if attack is not None:
            query = query.where(
                ResultRowRecord.attack.is_(None) if attack == "" else ResultRowRecord.attack == attack
            )
        if status is not None:
            query = query.where(ResultRowRecord.status == status)
        if file_name is not None:
            query = query.where(ResultRowRecord.file_name == file_name)
        total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
        records = session.scalars(query.order_by(ResultRowRecord.ordinal).offset(offset).limit(limit))
        return total, [(record.id, record.data) for record in records]


def get_row(run_id: uuid.UUID, row_id: int) -> dict[str, Any] | None:
    with session_scope() as session:
        record = session.get(ResultRowRecord, row_id)
        if record is None or record.run_id != run_id:
            return None
        return record.data


def all_rows(run_id: uuid.UUID) -> list[ExperimentResultRow]:
    with session_scope() as session:
        records = session.scalars(
            select(ResultRowRecord.data).where(ResultRowRecord.run_id == run_id).order_by(ResultRowRecord.ordinal)
        )
        return [ExperimentResultRow.model_validate(data) for data in records]


def row_facets(run_id: uuid.UUID) -> dict[str, list[Any]]:
    """Distinct values of the filterable columns of a run."""
    with session_scope() as session:
        def distinct(column):
            return [
                value
                for value in session.scalars(
                    select(column).where(ResultRowRecord.run_id == run_id).distinct().order_by(column)
                )
            ]

        return {
            "method": distinct(ResultRowRecord.method),
            "attack": distinct(ResultRowRecord.attack),
            "file_name": distinct(ResultRowRecord.file_name),
            "status": distinct(ResultRowRecord.status),
        }


def delete_run(run_id: uuid.UUID) -> bool:
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            return False
        session.execute(delete(ResultRowRecord).where(ResultRowRecord.run_id == run_id))
        session.delete(run)
        return True


def mark_interrupted_runs() -> int:
    """Runs left active by a previous server process can never finish."""
    with session_scope() as session:
        result = session.execute(
            update(Run)
            .where(Run.status.in_(ACTIVE_RUN_STATES))
            .values(status="interrupted", finished_at=_now(), error="The server stopped during the run.")
        )
        return result.rowcount or 0


def run_counts() -> dict[str, int]:
    with session_scope() as session:
        return dict(session.execute(select(Run.status, func.count()).group_by(Run.status)).all())


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def create_dataset(**fields: Any) -> Dataset:
    with session_scope() as session:
        dataset = Dataset(**fields)
        session.add(dataset)
        session.flush()
        return dataset


def update_dataset(dataset_id: uuid.UUID, **fields: Any) -> None:
    with session_scope() as session:
        session.execute(
            update(Dataset).where(Dataset.id == dataset_id).values(**{key: json_safe(value) for key, value in fields.items()})
        )


def get_dataset(dataset_id: uuid.UUID) -> Dataset | None:
    with session_scope() as session:
        return session.get(Dataset, dataset_id)


def list_datasets() -> list[Dataset]:
    with session_scope() as session:
        return list(session.scalars(select(Dataset).order_by(Dataset.created_at.desc())))


def delete_dataset(dataset_id: uuid.UUID) -> Dataset | None:
    with session_scope() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            return None
        session.delete(dataset)
        return dataset


def mark_interrupted_datasets() -> int:
    with session_scope() as session:
        result = session.execute(
            update(Dataset)
            .where(Dataset.status.in_(("pending", "downloading", "preparing")))
            .values(status="failed", error="The server stopped during preparation.")
        )
        return result.rowcount or 0


def as_uuid(value: str | uuid.UUID) -> uuid.UUID | None:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except ValueError:
        return None


__all__ = [
    "ACTIVE_RUN_STATES",
    "FINAL_RUN_STATES",
    "all_rows",
    "append_rows",
    "as_uuid",
    "create_dataset",
    "create_experiment",
    "create_run",
    "delete_dataset",
    "delete_experiment",
    "delete_run",
    "get_dataset",
    "get_experiment",
    "get_row",
    "json_safe",
    "get_run",
    "list_datasets",
    "list_experiments",
    "list_runs",
    "mark_interrupted_datasets",
    "mark_interrupted_runs",
    "query_rows",
    "row_facets",
    "run_counts",
    "update_dataset",
    "update_experiment",
    "update_run",
]
