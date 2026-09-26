"""The dataset library: prepare corpora, upload, register and generate audio."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from taf.corpora import SubsetRule
from taf.persistence import store
from taf.persistence.models import Dataset

from ..library import library
from ..schemas import DatasetDetail, DatasetOut, PrepareCorpusIn, RegisterLocalIn, SyntheticIn

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def dataset_out(dataset: Dataset) -> DatasetOut:
    return DatasetOut.model_validate(dataset, from_attributes=True)


@router.get("", response_model=list[DatasetOut])
def list_datasets() -> list[DatasetOut]:
    return [dataset_out(dataset) for dataset in store.list_datasets()]


@router.post("/prepare", response_model=DatasetOut, status_code=status.HTTP_202_ACCEPTED)
async def prepare_corpus(payload: PrepareCorpusIn) -> DatasetOut:
    """Download a standard corpus (or read a local copy) and prepare a subset in the background."""
    try:
        dataset = library.prepare_corpus(
            payload.corpus_id, payload.name, SubsetRule(**payload.rule.model_dump()), payload.source_path
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return dataset_out(dataset)


@router.post("/local", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def register_local(payload: RegisterLocalIn) -> DatasetOut:
    """Register a directory of audio files already on the server, without copying it."""
    try:
        dataset = library.register_local(**payload.model_dump())
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return dataset_out(dataset)


@router.post("/upload", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(files: list[UploadFile] = File(...), name: str = Form("Uploaded audio")) -> DatasetOut:
    payloads = [(upload.filename or "audio", await upload.read()) for upload in files]
    try:
        dataset = library.upload(name, payloads)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return dataset_out(dataset)


@router.post("/synthetic", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def create_synthetic(payload: SyntheticIn) -> DatasetOut:
    """Generate the deterministic set of synthetic edge-case signals."""
    return dataset_out(library.synthetic(payload.name, payload.sample_rate, payload.duration_seconds, payload.seed))


@router.get("/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: uuid.UUID) -> DatasetDetail:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return DatasetDetail.model_validate(dataset, from_attributes=True)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: uuid.UUID) -> None:
    if not library.delete(dataset_id):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
