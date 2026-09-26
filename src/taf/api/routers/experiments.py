"""Study protocols: create, edit, version and run experiments."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import ValidationError

from taf.experiments.runner import preview_experiment, validate_config
from taf.experiments.schema import ExperimentConfig
from taf.persistence import store
from taf.persistence.models import Experiment, Run

from ..runs import runs
from ..schemas import ExperimentIn, ExperimentOut, ExperimentPlan, RunBrief, RunOut

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

#: Fields of ``ExperimentConfig`` that describe the record, not the protocol.
_VOLATILE = {"experiment_id", "created_at", "created_by", "name", "experiment_type"}


def _stored_config(config: ExperimentConfig) -> dict:
    return store.json_safe(config.model_dump(mode="json", exclude=_VOLATILE))


def _problems(experiment: Experiment) -> list[str]:
    try:
        config = ExperimentConfig.model_validate(
            {**experiment.config, "name": experiment.name, "experiment_type": experiment.experiment_type}
        )
    except ValidationError as error:
        return [issue["msg"] for issue in error.errors()]
    return validate_config(config)


def run_brief(run: Run) -> RunBrief:
    return RunBrief.model_validate(run, from_attributes=True)


def experiment_out(experiment: Experiment, latest: Run | None = None, run_count: int = 0) -> ExperimentOut:
    return ExperimentOut(
        id=experiment.id,
        name=experiment.name,
        experiment_type=experiment.experiment_type,
        research_question=experiment.research_question,
        hypothesis=experiment.hypothesis,
        description=experiment.description,
        tags=list(experiment.tags or []),
        config=experiment.config,
        version=experiment.version,
        archived=experiment.archived,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        run_count=run_count,
        latest_run=run_brief(latest) if latest else None,
        problems=_problems(experiment),
    )


def _validated(payload: ExperimentIn) -> ExperimentConfig:
    try:
        return payload.experiment_config()
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=[{"loc": issue["loc"], "msg": issue["msg"]} for issue in error.errors()],
        ) from error


def _require(experiment_id: uuid.UUID) -> Experiment:
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
    return experiment


@router.post("/preview", response_model=ExperimentPlan)
def preview(config: ExperimentConfig) -> ExperimentPlan:
    """Dry run: counts, warnings and statistical-power notes for a configuration."""
    return preview_experiment(config)


@router.get("", response_model=list[ExperimentOut])
def list_experiments(include_archived: bool = Query(False)) -> list[ExperimentOut]:
    return [
        experiment_out(experiment, latest, count)
        for experiment, latest, count in store.list_experiments(include_archived)
    ]


@router.post("", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentIn) -> ExperimentOut:
    config = _validated(payload)
    experiment = store.create_experiment(
        name=payload.name,
        experiment_type=payload.experiment_type.value,
        research_question=payload.research_question,
        hypothesis=payload.hypothesis,
        description=payload.description,
        tags=payload.tags,
        config=_stored_config(config),
    )
    return experiment_out(experiment)


@router.get("/{experiment_id}", response_model=ExperimentOut)
def get_experiment(experiment_id: uuid.UUID) -> ExperimentOut:
    experiment = _require(experiment_id)
    runs_of = store.list_runs(experiment_id)
    latest = runs_of[0][0] if runs_of else None
    return experiment_out(experiment, latest, len(runs_of))


@router.put("/{experiment_id}", response_model=ExperimentOut)
def update_experiment(experiment_id: uuid.UUID, payload: ExperimentIn) -> ExperimentOut:
    _require(experiment_id)
    config = _validated(payload)
    experiment = store.update_experiment(
        experiment_id,
        name=payload.name,
        experiment_type=payload.experiment_type.value,
        research_question=payload.research_question,
        hypothesis=payload.hypothesis,
        description=payload.description,
        tags=payload.tags,
        config=_stored_config(config),
    )
    assert experiment is not None
    return get_experiment(experiment_id)


@router.post("/{experiment_id}/duplicate", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
def duplicate_experiment(experiment_id: uuid.UUID) -> ExperimentOut:
    source = _require(experiment_id)
    copy = store.create_experiment(
        name=f"{source.name} (copy)"[:200],
        experiment_type=source.experiment_type,
        research_question=source.research_question,
        hypothesis=source.hypothesis,
        description=source.description,
        tags=list(source.tags or []),
        config=dict(source.config),
    )
    return experiment_out(copy)


@router.post("/{experiment_id}/archive", response_model=ExperimentOut)
def archive_experiment(experiment_id: uuid.UUID, archived: bool = Query(True)) -> ExperimentOut:
    _require(experiment_id)
    store.update_experiment(experiment_id, archived=archived)
    return get_experiment(experiment_id)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(experiment_id: uuid.UUID) -> None:
    for run, _ in store.list_runs(experiment_id):
        runs.cancel(run.id)
    if not store.delete_experiment(experiment_id):
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")


@router.get("/{experiment_id}/runs", response_model=list[RunOut])
def list_experiment_runs(experiment_id: uuid.UUID) -> list[RunOut]:
    from .runs import run_out

    _require(experiment_id)
    return [run_out(run, experiment) for run, experiment in store.list_runs(experiment_id)]


@router.post("/{experiment_id}/runs", response_model=RunOut, status_code=status.HTTP_201_CREATED)
async def start_run(experiment_id: uuid.UUID) -> RunOut:
    from .runs import run_out

    try:
        run_id = runs.start(experiment_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, ValidationError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    run = store.get_run(run_id)
    experiment = store.get_experiment(experiment_id)
    assert run is not None and experiment is not None
    return run_out(run, experiment)
