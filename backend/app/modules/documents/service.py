import contextlib
import logging
import uuid
from collections.abc import Iterator

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    BadRequestException,
    ConflictException,
    NotFoundException,
    ServiceUnavailableException,
)
from app.modules.documents.models import Document, DocumentAsset, ProcessingStatus
from app.modules.documents.repository import DocumentRepository
from app.modules.documents.schemas import (
    DocumentAssetRead,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentProcessingStatusRead,
    DocumentRead,
    DocumentUpdateRequest,
)
from app.modules.documents.storage import (
    StorageProvider,
    build_storage_key,
    create_storage_provider,
)
from app.modules.documents.validation import sanitize_filename, validate_and_buffer_upload
from app.modules.processing.exceptions import DocumentProcessingError
from app.modules.processing.queue import ProcessingQueue, create_processing_queue

logger = logging.getLogger("app.modules.documents.service")


class DocumentService:
    def __init__(
        self,
        db: Session,
        *,
        storage: StorageProvider | None = None,
        processing_queue: ProcessingQueue | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.repo = DocumentRepository(db)
        self.settings = settings or get_settings()
        self.storage = storage or create_storage_provider(self.settings)
        self.processing_queue = processing_queue or create_processing_queue(self.settings)

    def upload_document(
        self,
        user_id: uuid.UUID,
        upload_file: UploadFile,
    ) -> DocumentRead:
        validated = validate_and_buffer_upload(
            upload_file,
            max_size_bytes=self.settings.MAX_UPLOAD_SIZE_BYTES,
        )

        document_id = uuid.uuid4()
        storage_key = build_storage_key(user_id, document_id, validated.file_extension)

        try:
            self.storage.save(
                storage_key=storage_key,
                stream=validated.stream,
                content_type=validated.mime_type,
            )
        finally:
            validated.close()

        try:
            doc = self.repo.create_document(
                document_id=document_id,
                user_id=user_id,
                original_filename=validated.original_filename,
                storage_key=storage_key,
                file_extension=validated.file_extension,
                mime_type=validated.mime_type,
                file_size_bytes=validated.file_size_bytes,
                checksum_sha256=validated.checksum_sha256,
            )
            doc.processing_status = ProcessingStatus.QUEUED.value
            self.db.commit()
            self.db.refresh(doc)
        except Exception as exc:
            self.db.rollback()
            # Ensure no orphan file remains in storage when DB persistence fails
            try:
                self.storage.delete(storage_key)
            except Exception:
                logger.exception(
                    "Failed to clean up storage object after DB rollback",
                    extra={"extra_data": {"document_id": str(document_id)}},
                )
            logger.exception(
                "Database error while recording uploaded document",
                extra={"extra_data": {"error_type": type(exc).__name__}},
            )
            raise ServiceUnavailableException(
                "Could not save document metadata",
                code="DOCUMENT_SAVE_FAILED",
            ) from exc

        self._enqueue_on_upload(doc)

        logger.info(
            "Document uploaded",
            extra={
                "extra_data": {
                    "user_id": str(user_id),
                    "document_id": str(doc.id),
                    "file_extension": doc.file_extension,
                    "file_size_bytes": doc.file_size_bytes,
                    "processing_status": doc.processing_status,
                }
            },
        )
        return DocumentRead.model_validate(doc)

    def _enqueue_on_upload(self, doc: Document) -> None:
        try:
            self.processing_queue.enqueue_document(doc.id)
            self.db.refresh(doc)
        except DocumentProcessingError as exc:
            doc.processing_status = ProcessingStatus.FAILED.value
            doc.processing_error_code = exc.code[:64]
            doc.processing_error_message = exc.user_message[:500]
            self.db.commit()
            self.db.refresh(doc)
        except Exception as exc:
            logger.warning(
                "Unexpected error enqueueing uploaded document for processing",
                extra={
                    "extra_data": {
                        "document_id": str(doc.id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            doc.processing_status = ProcessingStatus.FAILED.value
            doc.processing_error_code = "QUEUE_UNAVAILABLE"
            doc.processing_error_message = (
                "Background processing queue is temporarily unavailable. "
                "Please tap Retry Processing."
            )
            self.db.commit()
            self.db.refresh(doc)

    def get_processing_status(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> DocumentProcessingStatusRead:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")
        return DocumentProcessingStatusRead.model_validate(doc)

    def retry_processing(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> DocumentRead:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")

        if doc.processing_status == ProcessingStatus.PROCESSING.value:
            raise ConflictException(
                "Document is currently being processed",
                code="DOCUMENT_ALREADY_PROCESSING",
            )
        if doc.processing_status == ProcessingStatus.READY.value:
            raise ConflictException(
                "Document has already been processed and is ready",
                code="DOCUMENT_ALREADY_READY",
            )

        doc.processing_status = ProcessingStatus.QUEUED.value
        doc.processing_error_code = None
        doc.processing_error_message = None
        self.db.commit()

        try:
            self.processing_queue.enqueue_document(doc.id)
            self.db.refresh(doc)
        except DocumentProcessingError as exc:
            doc.processing_status = ProcessingStatus.FAILED.value
            doc.processing_error_code = exc.code[:64]
            doc.processing_error_message = exc.user_message[:500]
            self.db.commit()
            raise ServiceUnavailableException(
                exc.user_message,
                code=exc.code,
            ) from exc

        logger.info(
            "Document processing retry queued",
            extra={
                "extra_data": {
                    "user_id": str(user_id),
                    "document_id": str(doc.id),
                    "processing_status": doc.processing_status,
                }
            },
        )
        return DocumentRead.model_validate(doc)

    def list_documents(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> DocumentListResponse:
        items, total = self.repo.list_documents_for_user(
            user_id,
            limit=limit,
            offset=offset,
        )
        return DocumentListResponse(
            items=[DocumentRead.model_validate(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_document(self, user_id: uuid.UUID, document_id: uuid.UUID) -> DocumentRead:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")
        return DocumentRead.model_validate(doc)

    def update_document(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        payload: DocumentUpdateRequest,
    ) -> DocumentRead:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")

        fields_set = payload.model_fields_set

        if "original_filename" in fields_set and payload.original_filename is not None:
            raw_name = payload.original_filename
            if "." not in raw_name:
                raw_name = f"{raw_name}.{doc.file_extension}"
            safe_filename, ext = sanitize_filename(raw_name)
            if ext != doc.file_extension:
                raise BadRequestException(
                    f"Cannot change document extension from '.{doc.file_extension}' to '.{ext}'",
                    code="EXTENSION_CHANGE_NOT_ALLOWED",
                )
            doc.original_filename = safe_filename

        self.db.commit()
        self.db.refresh(doc)
        return DocumentRead.model_validate(doc)

    def delete_document(self, user_id: uuid.UUID, document_id: uuid.UUID) -> DocumentDeleteResponse:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")

        # Clean up any stored visual assets
        for asset in doc.assets:
            with contextlib.suppress(Exception):
                self.storage.delete(asset.storage_key)

        storage_key = doc.storage_key
        # Delete from storage first so a transient storage failure does not orphan the blob
        self.storage.delete(storage_key)
        self.repo.delete_document(doc)
        self.db.commit()

        logger.info(
            "Document deleted",
            extra={
                "extra_data": {
                    "user_id": str(user_id),
                    "document_id": str(document_id),
                }
            },
        )
        return DocumentDeleteResponse(deleted=True)

    def open_document_download(
        self, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> tuple[Document, Iterator[bytes]]:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")
        stream = self.storage.open_stream(doc.storage_key)
        return doc, stream

    def list_document_assets(
        self, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> list[DocumentAssetRead]:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")
        stmt = (
            select(DocumentAsset)
            .where(DocumentAsset.document_id == document_id, DocumentAsset.user_id == user_id)
            .order_by(DocumentAsset.order_index.asc())
        )
        assets = self.db.scalars(stmt).all()
        return [DocumentAssetRead.model_validate(a) for a in assets]

    def open_asset_download(
        self, user_id: uuid.UUID, document_id: uuid.UUID, asset_id: uuid.UUID
    ) -> tuple[DocumentAsset, Iterator[bytes]]:
        doc = self.repo.get_document_for_user(document_id, user_id)
        if doc is None:
            raise NotFoundException("Document not found", code="DOCUMENT_NOT_FOUND")
        stmt = select(DocumentAsset).where(
            DocumentAsset.id == asset_id,
            DocumentAsset.document_id == document_id,
            DocumentAsset.user_id == user_id,
        )
        asset = self.db.scalar(stmt)
        if asset is None:
            raise NotFoundException("Document asset not found", code="ASSET_NOT_FOUND")
        stream = self.storage.open_stream(asset.storage_key)
        return asset, stream
