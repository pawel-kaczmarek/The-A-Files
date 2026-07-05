from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..schemas import UploadedDatasetInfo
from ..uploads import UploadedDataset, upload_registry

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/uploads", response_model=UploadedDatasetInfo, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    files: list[UploadFile] = File(...),
    name: str | None = Form(default=None),
) -> UploadedDatasetInfo:
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        payloads.append((upload.filename or "audio.wav", await upload.read()))
    try:
        dataset = upload_registry.create(name, payloads)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _to_info(dataset)


@router.get("/uploads", response_model=list[UploadedDatasetInfo])
def list_uploads() -> list[UploadedDatasetInfo]:
    return [_to_info(dataset) for dataset in upload_registry.list()]


@router.delete("/uploads/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_upload(upload_id: str) -> None:
    if not upload_registry.delete(upload_id):
        raise HTTPException(status_code=404, detail=f"Uploaded dataset not found: {upload_id}")


def _to_info(dataset: UploadedDataset) -> UploadedDatasetInfo:
    return UploadedDatasetInfo(
        id=dataset.id,
        name=dataset.name,
        dataset_name=dataset.dataset_name,
        file_count=len(dataset.files),
        file_names=[path.name for path in dataset.files],
        created_at=dataset.created_at,
    )
