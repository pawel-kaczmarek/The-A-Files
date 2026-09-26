"""Relational model of the research platform.

* An **experiment** is a study protocol: research question, hypothesis,
  design and configuration. Editing its configuration increments its
  version, so every run can be traced to the exact protocol it executed.
* A **run** is one execution of an experiment. It stores the configuration
  as resolved at start (with the drawn seed), the run manifest (versions,
  input digests) and the scenario summary.
* A **result row** is one trial of a run. The full normalized row is kept as
  JSONB; the columns used to filter and group are duplicated as indexed
  columns.
* A **dataset** is audio material in the platform library: a prepared
  subset of a standard corpus, an upload, a registered local directory or a
  synthetic test set, with the manifest that pins its content.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    experiment_type: Mapped[str] = mapped_column(String(40), index=True)
    research_question: Mapped[str | None] = mapped_column(Text)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[Any]] = mapped_column(default=list)
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    runs: Mapped[list["Run"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="Run.number"
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    number: Mapped[int] = mapped_column(Integer)
    experiment_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(default=dict)
    manifest: Mapped[dict[str, Any]] = mapped_column(default=dict)
    plan: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    completed_rows: Mapped[int] = mapped_column(Integer, default=0)
    ok_rows: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    rows: Mapped[list["ResultRowRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class ResultRowRecord(Base):
    __tablename__ = "result_rows"
    __table_args__ = (
        Index("ix_result_rows_run_method", "run_id", "method"),
        Index("ix_result_rows_run_attack", "run_id", "attack"),
        Index("ix_result_rows_run_ordinal", "run_id", "ordinal"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    file_name: Mapped[str] = mapped_column(String(400))
    method: Mapped[str] = mapped_column(String(300))
    method_name: Mapped[str | None] = mapped_column(String(100))
    payload_length: Mapped[int] = mapped_column(Integer)
    repetition: Mapped[int] = mapped_column(Integer)
    attack: Mapped[str | None] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(20))
    failure_kind: Mapped[str | None] = mapped_column(String(40))
    ber: Mapped[float | None] = mapped_column(Float)
    decode_success: Mapped[bool] = mapped_column(Boolean)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)

    run: Mapped[Run] = relationship(back_populates="rows")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    #: "corpus" (prepared subset), "upload", "local" or "synthetic".
    kind: Mapped[str] = mapped_column(String(20), index=True)
    corpus_id: Mapped[str | None] = mapped_column(String(60))
    #: "pending", "downloading", "preparing", "ready" or "failed".
    status: Mapped[str] = mapped_column(String(20), default="pending")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(20))
    path: Mapped[str | None] = mapped_column(Text)
    domain: Mapped[str | None] = mapped_column(String(40))
    language: Mapped[str | None] = mapped_column(String(20))
    license: Mapped[str | None] = mapped_column(String(200))
    citation: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_seconds: Mapped[float | None] = mapped_column(Float)
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    rule: Mapped[dict[str, Any]] = mapped_column(default=dict)
    manifest: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


__all__ = ["Base", "Dataset", "Experiment", "ResultRowRecord", "Run"]
