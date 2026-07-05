"""API-facing models. The experiment schema itself lives in ``taf.experiments``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from taf.experiments.results import ExperimentResultRow
from taf.experiments.schema import ExperimentConfig, ExperimentPlan, ExperimentType

__all__ = [
    "AttackInfo",
    "AttackParameterInfo",
    "CatalogSummary",
    "DatasetInfo",
    "ExperimentConfig",
    "ExperimentDetail",
    "ExperimentPlan",
    "ExperimentResultRow",
    "ExperimentType",
    "JobSummary",
    "MethodInfo",
    "MetricInfo",
    "UploadedDatasetInfo",
]


class MethodInfo(BaseModel):
    name: str
    class_name: str
    description: str
    requires_tensorflow: bool = False
    needs_long_input: bool = False


class MetricInfo(BaseModel):
    name: str
    class_name: str
    category: str
    requires_tensorflow: bool = False
    compares_original: bool = True
    supports_attacked_audio: bool = True


class AttackParameterInfo(BaseModel):
    name: str
    default: Any = None


class AttackInfo(BaseModel):
    name: str
    class_name: str
    description: str = ""
    parameters: list[AttackParameterInfo] = Field(default_factory=list)
    changes_length_or_rate: bool = False


class DatasetInfo(BaseModel):
    id: str
    label: str
    kind: Literal["packaged", "uploaded"]
    file_count: int
    formats: list[str] = Field(default_factory=list)


class UploadedDatasetInfo(BaseModel):
    id: str
    name: str
    dataset_name: str
    file_count: int
    file_names: list[str]
    created_at: datetime


class CatalogSummary(BaseModel):
    methods: int
    metrics: int
    attacks: int
    datasets: int


class JobSummary(BaseModel):
    """Status snapshot of one experiment run held by the API."""

    experiment_id: str
    experiment_type: ExperimentType
    name: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    dataset: str | None = None
    methods: list[str]
    metrics: list[str]
    attacks: list[str]
    payload_lengths: list[int]
    repetitions: int
    total_tasks: int
    completed_rows: int
    success_rows: int
    error: str | None = None
    csv_url: str
    summary_csv_url: str
    config: ExperimentConfig


class ExperimentDetail(JobSummary):
    rows: list[ExperimentResultRow] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
