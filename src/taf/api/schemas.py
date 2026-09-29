"""API-facing models. The experiment schema itself lives in ``taf.experiments``."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from taf.experiments.results import ExperimentResultRow
from taf.experiments.schema import ExperimentConfig, ExperimentPlan, ExperimentType

# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------


class LocalizedText(BaseModel):
    """A text in every interface language; missing translations fall back to English."""

    en: str = ""
    pl: str = ""


class ReferenceInfo(BaseModel):
    citation: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    #: DOI resolver link, or the URL.
    link: str | None = None


class ComponentCardInfo(BaseModel):
    """Fields every method, metric and attack takes from its card (``taf.models.card``)."""

    title: LocalizedText = Field(default_factory=LocalizedText)
    #: One sentence: what the component does, models or measures.
    summary: LocalizedText = Field(default_factory=LocalizedText)
    #: Mechanism, parameters, interpretation and departures from the publication.
    details: LocalizedText = Field(default_factory=LocalizedText)
    #: Short name for figures and tables ("QIM", "DCT-b1").
    abbreviation: str = ""
    references: list[ReferenceInfo] = Field(default_factory=list)
    #: Importable modules needed beyond the core installation.
    requires: list[str] = Field(default_factory=list)
    #: Optional-dependency group that installs them.
    extra: str | None = None
    #: Whether the requirements are installed on this server.
    available: bool = True
    #: First reference, flattened.
    reference: str | None = None
    year: int | None = None
    doi: str | None = None


class MethodParameterInfo(BaseModel):
    name: str
    default: Any = None
    type: str | None = None
    required: bool = False
    is_key: bool = False


class MethodInfo(ComponentCardInfo):
    name: str
    class_name: str
    #: The method's own label (``type()``), as recorded in result rows.
    description: str
    #: False for methods registered by a plugin distribution.
    packaged: bool = True
    family: str | None = None
    family_label: LocalizedText = Field(default_factory=LocalizedText)
    purpose: str | None = None
    purpose_label: LocalizedText | None = None
    strength_parameter: str | None = None
    parameters: list[MethodParameterInfo] = Field(default_factory=list)
    requires_tensorflow: bool = False
    needs_long_input: bool = False


class MetricInfo(ComponentCardInfo):
    domain: str | None = None
    name: str
    #: The metric's own label (``name()``), as recorded in result rows.
    label: str = ""
    class_name: str
    category: str
    category_label: LocalizedText = Field(default_factory=LocalizedText)
    packaged: bool = True
    requires_tensorflow: bool = False
    #: True: higher means closer to the original; False: lower does.
    higher_is_better: bool | None = None
    #: Named entries of a multi-valued result, reported separately.
    components: list[str] = Field(default_factory=list)
    scale: str | None = None
    intrusive: bool = True
    compares_original: bool = True
    supports_attacked_audio: bool = True


class AttackParameterInfo(BaseModel):
    name: str
    default: Any = None


class AttackInfo(ComponentCardInfo):
    name: str
    class_name: str
    #: English summary, kept for older clients.
    description: str = ""
    family: str = ""
    family_label: LocalizedText = Field(default_factory=LocalizedText)
    parameters: list[AttackParameterInfo] = Field(default_factory=list)
    changes_length_or_rate: bool = False
    stochastic: bool = False
    has_severity: bool = False
    sweep: dict[str, Any] | None = None


class DesignInfo(BaseModel):
    type: ExperimentType
    title: str
    description: str
    property: str
    factors: list[str]
    measures: list[str]
    analyses: list[str]
    requires_metrics: bool
    requires_attacks: bool
    requires_sweep: str | None = None
    min_methods: int = 1
    default_payload_lengths: list[int]


class CatalogDataset(BaseModel):
    """A dataset an experiment can run on: packaged or from the library."""

    id: str
    label: str
    kind: str
    file_count: int
    domain: str | None = None
    sample_rate: int | None = None
    total_duration_seconds: float | None = None


# --------------------------------------------------------------------------
# Experiments and runs
# --------------------------------------------------------------------------


class ExperimentIn(BaseModel):
    """A study protocol. ``config`` holds the ``ExperimentConfig`` fields;
    its name and type are taken from the top level."""

    name: str = Field(min_length=1, max_length=200)
    experiment_type: ExperimentType
    research_question: str | None = None
    hypothesis: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any]

    def experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {**self.config, "name": self.name, "experiment_type": self.experiment_type}
        )


class RunBrief(BaseModel):
    id: uuid.UUID
    number: int
    status: str
    experiment_version: int
    total_rows: int
    completed_rows: int
    ok_rows: int
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExperimentOut(BaseModel):
    id: uuid.UUID
    name: str
    experiment_type: ExperimentType
    research_question: str | None = None
    hypothesis: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any]
    version: int
    archived: bool
    created_at: datetime
    updated_at: datetime
    run_count: int = 0
    latest_run: RunBrief | None = None
    #: Scenario-level problems that would stop a run (empty when runnable).
    problems: list[str] = Field(default_factory=list)


class RunOut(RunBrief):
    experiment_id: uuid.UUID
    experiment_name: str
    experiment_type: ExperimentType
    config: dict[str, Any]
    plan: dict[str, Any] = Field(default_factory=dict)


class RowRecord(BaseModel):
    id: int
    row: ExperimentResultRow


class RowsPage(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[RowRecord]


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


class DatasetOut(BaseModel):
    id: uuid.UUID
    name: str
    kind: str
    corpus_id: str | None = None
    status: str
    progress: float = 0.0
    stage: str | None = None
    path: str | None = None
    domain: str | None = None
    language: str | None = None
    license: str | None = None
    citation: str | None = None
    description: str | None = None
    file_count: int = 0
    total_duration_seconds: float | None = None
    sample_rate: int | None = None
    rule: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime


class DatasetDetail(DatasetOut):
    manifest: dict[str, Any] = Field(default_factory=dict)


class SubsetRuleIn(BaseModel):
    max_files: int = Field(default=100, ge=1, le=5000)
    target_sample_rate: int | None = Field(default=16000, ge=8000, le=96000)
    min_duration_seconds: float = Field(default=1.0, ge=0)
    max_duration_seconds: float | None = Field(default=30.0, gt=0)
    excerpt_seconds: float | None = Field(default=None, gt=0)
    excerpt_offset_seconds: float = Field(default=0.0, ge=0)
    seed: int = 0


class PrepareCorpusIn(BaseModel):
    corpus_id: str
    name: str | None = None
    rule: SubsetRuleIn = Field(default_factory=SubsetRuleIn)
    #: Local archive or directory to prepare from instead of downloading
    #: (for licensed corpora and for archives already on disk).
    source_path: str | None = None


class RegisterLocalIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    path: str
    corpus_id: str | None = None
    domain: str | None = None
    license: str | None = None
    citation: str | None = None
    description: str | None = None


class SyntheticIn(BaseModel):
    name: str | None = None
    sample_rate: int = Field(default=16000, ge=8000, le=96000)
    duration_seconds: float = Field(default=5.0, gt=0, le=60)
    seed: int = 0


class SystemStatus(BaseModel):
    status: str
    version: str
    database: dict[str, Any]
    data_dir: str


class PlatformStats(BaseModel):
    experiments: int
    runs: dict[str, int]
    datasets: int
    methods: int
    metrics: int
    attacks: int
    designs: int


__all__ = [
    "AttackInfo",
    "AttackParameterInfo",
    "CatalogDataset",
    "DatasetDetail",
    "DatasetOut",
    "DesignInfo",
    "ExperimentConfig",
    "ExperimentIn",
    "ExperimentOut",
    "ExperimentPlan",
    "ExperimentResultRow",
    "ExperimentType",
    "MethodInfo",
    "MethodParameterInfo",
    "MetricInfo",
    "PlatformStats",
    "PrepareCorpusIn",
    "RegisterLocalIn",
    "RowRecord",
    "RowsPage",
    "RunBrief",
    "RunOut",
    "SubsetRuleIn",
    "SyntheticIn",
    "SystemStatus",
]
