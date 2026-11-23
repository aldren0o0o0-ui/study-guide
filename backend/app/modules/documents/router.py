import uuid
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.documents.schemas import (
    DocumentAssetRead,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentProcessingStatusRead,
    DocumentRead,
    DocumentUpdateRequest,
)
from app.modules.documents.service import DocumentService
from app.modules.documents.storage import StorageProvider, get_storage_provider
from app.modules.processing.queue import ProcessingQueue, get_processing_queue
from app.shared.responses import ApiResponse

documents_router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(
    db: Annotated[Session, Depends(get_db)],
    storage: Annotated[StorageProvider, Depends(get_storage_provider)],
    processing_queue: Annotated[ProcessingQueue, Depends(get_processing_queue)],
) -> DocumentService:
    return DocumentService(db, storage=storage, processing_queue=processing_queue)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


@documents_router.post(
    "",
    response_model=ApiResponse[DocumentRead],
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    current_user: CurrentUser,
    service: DocumentServiceDep,
    file: Annotated[UploadFile, File(...)],
) -> ApiResponse[DocumentRead]:
    document = service.upload_document(current_user.id, file)
    return ApiResponse(data=document, message="Document uploaded")


@documents_router.get(
    "",
    response_model=ApiResponse[DocumentListResponse],
    status_code=status.HTTP_200_OK,
)
def list_documents(
    current_user: CurrentUser,
    service: DocumentServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[DocumentListResponse]:
    result = service.list_documents(
        current_user.id,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(data=result)


@documents_router.get(
    "/{document_id}",
    response_model=ApiResponse[DocumentRead],
    status_code=status.HTTP_200_OK,
)
def get_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> ApiResponse[DocumentRead]:
    document = service.get_document(current_user.id, document_id)
    return ApiResponse(data=document)


@documents_router.get(
    "/{document_id}/processing-status",
    response_model=ApiResponse[DocumentProcessingStatusRead],
    status_code=status.HTTP_200_OK,
)
def get_document_processing_status(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> ApiResponse[DocumentProcessingStatusRead]:
    status_data = service.get_processing_status(current_user.id, document_id)
    return ApiResponse(data=status_data)


@documents_router.post(
    "/{document_id}/retry-processing",
    response_model=ApiResponse[DocumentRead],
    status_code=status.HTTP_200_OK,
)
def retry_document_processing(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> ApiResponse[DocumentRead]:
    document = service.retry_processing(current_user.id, document_id)
    return ApiResponse(data=document, message="Document processing retry queued")


@documents_router.patch(
    "/{document_id}",
    response_model=ApiResponse[DocumentRead],
    status_code=status.HTTP_200_OK,
)
def update_document(
    document_id: uuid.UUID,
    payload: DocumentUpdateRequest,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> ApiResponse[DocumentRead]:
    document = service.update_document(current_user.id, document_id, payload)
    return ApiResponse(data=document, message="Document updated")


@documents_router.delete(
    "/{document_id}",
    response_model=ApiResponse[DocumentDeleteResponse],
    status_code=status.HTTP_200_OK,
)
def delete_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> ApiResponse[DocumentDeleteResponse]:
    result = service.delete_document(current_user.id, document_id)
    return ApiResponse(data=result, message="Document deleted")


@documents_router.get(
    "/{document_id}/download",
    status_code=status.HTTP_200_OK,
)
def download_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> StreamingResponse:
    doc, stream = service.open_document_download(current_user.id, document_id)
    encoded_filename = quote(doc.original_filename)
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
        "X-Content-Type-Options": "nosniff",
        "Content-Length": str(doc.file_size_bytes),
    }
    return StreamingResponse(
        stream,
        media_type=doc.mime_type,
        headers=headers,
    )


@documents_router.get(
    "/{document_id}/assets",
    response_model=ApiResponse[list[DocumentAssetRead]],
    status_code=status.HTTP_200_OK,
)
def list_document_assets(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> ApiResponse[list[DocumentAssetRead]]:
    assets = service.list_document_assets(current_user.id, document_id)
    return ApiResponse(data=assets)


@documents_router.get(
    "/{document_id}/assets/{asset_id}",
    status_code=status.HTTP_200_OK,
)
def download_document_asset(
    document_id: uuid.UUID,
    asset_id: uuid.UUID,
    current_user: CurrentUser,
    service: DocumentServiceDep,
) -> StreamingResponse:
    asset, stream = service.open_asset_download(current_user.id, document_id, asset_id)
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Content-Length": str(asset.file_size_bytes),
    }
    return StreamingResponse(
        stream,
        media_type=asset.mime_type,
        headers=headers,
    )
