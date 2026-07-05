from __future__ import annotations

from fastapi import APIRouter

from taf.experiments import registry as experiment_registry

from ..schemas import AttackInfo, AttackParameterInfo, CatalogSummary, DatasetInfo, MethodInfo, MetricInfo

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("/methods", response_model=list[MethodInfo])
def list_methods() -> list[MethodInfo]:
    return [MethodInfo(**row) for row in experiment_registry.list_methods()]


@router.get("/metrics", response_model=list[MetricInfo])
def list_metrics() -> list[MetricInfo]:
    return [MetricInfo(**row) for row in experiment_registry.list_metrics()]


@router.get("/attacks", response_model=list[AttackInfo])
def list_attacks() -> list[AttackInfo]:
    return [
        AttackInfo(
            name=spec.name,
            class_name=spec.class_name,
            description=spec.description,
            parameters=[
                AttackParameterInfo(name=param.name, default=param.default)
                for param in spec.parameters
            ],
            changes_length_or_rate=spec.changes_length_or_rate,
        )
        for spec in experiment_registry.list_attacks()
    ]


@router.get("/datasets", response_model=list[DatasetInfo])
def list_datasets() -> list[DatasetInfo]:
    return [DatasetInfo(**row) for row in experiment_registry.list_datasets()]


@router.get("/summary", response_model=CatalogSummary)
def catalog_summary() -> CatalogSummary:
    return CatalogSummary(
        methods=len(experiment_registry.list_methods()),
        metrics=len(experiment_registry.list_metrics()),
        attacks=len(experiment_registry.list_attacks()),
        datasets=len(experiment_registry.list_datasets()),
    )
