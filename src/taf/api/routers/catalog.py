"""What the platform can evaluate: methods, metrics, attacks, designs, corpora."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Query

from taf.experiments import registry as experiment_registry
from taf.persistence import store

from ..schemas import AttackInfo, AttackParameterInfo, CatalogDataset, DesignInfo, MethodInfo, MetricInfo

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


# The packaged catalogue does not change while the server runs; building it
# instantiates every method and metric, so it is computed once.
@lru_cache(maxsize=1)
def _methods() -> list[MethodInfo]:
    return [MethodInfo(**row) for row in experiment_registry.list_methods()]


@lru_cache(maxsize=1)
def _metrics() -> list[MetricInfo]:
    return [MetricInfo(**row) for row in experiment_registry.list_metrics()]


@lru_cache(maxsize=1)
def _attacks() -> list[AttackInfo]:
    return [
        AttackInfo(
            name=spec.name,
            class_name=spec.class_name,
            description=spec.description,
            family=spec.family,
            parameters=[AttackParameterInfo(name=p.name, default=p.default) for p in spec.parameters],
            changes_length_or_rate=spec.changes_length_or_rate,
            stochastic=spec.stochastic,
            has_severity=spec.has_severity,
            sweep=spec.sweep,
        )
        for spec in experiment_registry.list_attacks()
    ]


@router.get("/methods", response_model=list[MethodInfo])
def list_methods() -> list[MethodInfo]:
    return _methods()


@router.get("/metrics", response_model=list[MetricInfo])
def list_metrics() -> list[MetricInfo]:
    return _metrics()


@router.get("/attacks", response_model=list[AttackInfo])
def list_attacks() -> list[AttackInfo]:
    return _attacks()


@router.get("/designs", response_model=list[DesignInfo])
def list_designs() -> list[DesignInfo]:
    return [DesignInfo(**row) for row in experiment_registry.list_designs()]


@router.get("/presets")
def attack_presets(sample_rate: int = Query(16000, ge=8000, le=96000)) -> dict[str, Any]:
    """Benchmark suites, sweep ladders and pipelines resolved for a sampling rate."""
    return experiment_registry.attack_presets(sample_rate)


@router.get("/corpora")
def list_corpora() -> list[dict[str, Any]]:
    from taf.corpora import CORPORA

    return [corpus.describe() for corpus in CORPORA]


@router.get("/datasets", response_model=list[CatalogDataset])
def list_datasets() -> list[CatalogDataset]:
    """Datasets an experiment can use: packaged sets and ready library entries."""
    packaged = [
        CatalogDataset(
            id=row["id"], label=row["label"], kind="packaged", file_count=row["file_count"], domain="speech"
        )
        for row in experiment_registry.list_datasets()
    ]
    library = [
        CatalogDataset(
            id=f"library:{dataset.id}",
            label=dataset.name,
            kind=dataset.kind,
            file_count=dataset.file_count,
            domain=dataset.domain,
            sample_rate=dataset.sample_rate,
            total_duration_seconds=dataset.total_duration_seconds,
        )
        for dataset in store.list_datasets()
        if dataset.status == "ready"
    ]
    return packaged + library
