import io
import logging
import uuid
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.modules.documents.models import Document, DocumentAsset, DocumentChunk, ProcessingStatus
from app.modules.documents.storage import (
    StorageProvider,
    build_asset_storage_key,
    create_storage_provider,
)
from app.modules.processing.chunking import ChunkingService, TextChunk
from app.modules.processing.embeddings import EmbeddingProvider, create_embedding_provider
from app.modules.processing.exceptions import DocumentProcessingError
from app.modules.processing.extraction import ExtractedAsset, extract_document_content

logger = logging.getLogger("app.modules.processing.service")


class DocumentProcessingService:
    """
    Executes the end-to-end document processing pipeline:
    1. Storage read
    2. Text extraction & normalization
    3. Deterministic chunking
    4. Gemini embedding generation
    5. Atomic chunk persistence & status transition
    """

    def __init__(
        self,
        db: Session,
        *,
        storage: StorageProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        chunker: ChunkingService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.storage = storage or create_storage_provider(self.settings)
        self.embedding_provider = embedding_provider or create_embedding_provider(self.settings)
        self.chunker = chunker or ChunkingService(
            chunk_size=self.settings.CHUNK_SIZE_CHARS,
            chunk_overlap=self.settings.CHUNK_OVERLAP_CHARS,
        )

    def process_document(self, document_id: uuid.UUID) -> Document | None:
        doc = self._acquire_document_for_processing(document_id)
        if doc is None:
            return None

        if doc.processing_status == ProcessingStatus.PROCESSING.value:
            self.db.commit()
            logger.info(
                "Skipping duplicate job execution; document is already processing",
                extra={"extra_data": {"document_id": str(document_id)}},
            )
            return doc

        # Mark as processing and clear stale errors
        doc.processing_status = ProcessingStatus.PROCESSING.value
        doc.processing_error_code = None
        doc.processing_error_message = None
        self.db.commit()

        try:
            stream = self.storage.open_stream(doc.storage_key)
            extracted = extract_document_content(
                stream,
                file_extension=doc.file_extension,
                max_size_bytes=self.settings.MAX_UPLOAD_SIZE_BYTES,
            )
            chunks = self.chunker.chunk_document(extracted)
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_provider.embed_documents(chunk_texts)

            if len(embeddings) != len(chunks):
                raise DocumentProcessingError(
                    f"Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})",
                    code="EMBEDDING_COUNT_MISMATCH",
                )

            return self._persist_chunks_and_mark_ready(
                document_id=document_id,
                chunks=chunks,
                embeddings=embeddings,
                assets=extracted.assets,
            )
        except DocumentProcessingError as exc:
            return self._record_failure(
                document_id=document_id,
                error_code=exc.code,
                user_message=exc.user_message,
                exc=exc,
            )
        except AppException as exc:
            return self._record_failure(
                document_id=document_id,
                error_code=exc.code,
                user_message=exc.message,
                exc=exc,
            )
        except Exception as exc:
            return self._record_failure(
                document_id=document_id,
                error_code="INTERNAL_PROCESSING_ERROR",
                user_message=(
                    "An unexpected error occurred while processing this document. Please try again."
                ),
                exc=exc,
            )

    def _acquire_document_for_processing(self, document_id: uuid.UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id).with_for_update()
        doc = self.db.scalar(stmt)
        if doc is None:
            self.db.rollback()
            logger.warning(
                "Document not found when starting processing job (may have been deleted)",
                extra={"extra_data": {"document_id": str(document_id)}},
            )
            return None
        return doc

    def _persist_chunks_and_mark_ready(
        self,
        *,
        document_id: uuid.UUID,
        chunks: Sequence[TextChunk],
        embeddings: list[list[float]],
        assets: Sequence[ExtractedAsset] = (),
    ) -> Document | None:
        stmt = select(Document).where(Document.id == document_id).with_for_update()
        doc = self.db.scalar(stmt)
        if doc is None:
            self.db.rollback()
            logger.warning(
                "Document was deleted during processing; discarding generated chunks",
                extra={"extra_data": {"document_id": str(document_id)}},
            )
            return None

        # Atomically replace any existing chunks for this document
        self.db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        self.db.flush()

        for chunk, vector in zip(chunks, embeddings, strict=True):
            db_chunk = DocumentChunk(
                document_id=doc.id,
                user_id=doc.user_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_hash=chunk.content_hash,
                char_count=chunk.char_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                embedding=vector,
            )
            self.db.add(db_chunk)

        # Replace any existing visual assets for this document
        existing_assets = self.db.scalars(
            select(DocumentAsset).where(DocumentAsset.document_id == doc.id)
        ).all()
        for old_asset in existing_assets:
            with suppress(Exception):
                self.storage.delete(old_asset.storage_key)
        self.db.execute(delete(DocumentAsset).where(DocumentAsset.document_id == doc.id))
        self.db.flush()

        # Persist extracted visual assets safely
        for asset in assets:
            try:
                asset_id = uuid.uuid4()
                storage_key = build_asset_storage_key(
                    doc.user_id,
                    doc.id,
                    asset_id,
                    asset.file_extension,
                )
                self.storage.save(storage_key, io.BytesIO(asset.data), asset.mime_type)
                db_asset = DocumentAsset(
                    id=asset_id,
                    document_id=doc.id,
                    user_id=doc.user_id,
                    storage_key=storage_key,
                    asset_type="image",
                    mime_type=asset.mime_type,
                    file_size_bytes=len(asset.data),
                    checksum_sha256=asset.checksum_sha256,
                    page_number=asset.page_number,
                    section_title=asset.section_title,
                    caption=asset.caption,
                    order_index=asset.order_index,
                    width=asset.width,
                    height=asset.height,
                )
                self.db.add(db_asset)
            except Exception as asset_exc:
                logger.warning(
                    "Failed to save extracted asset for document %s: %s",
                    doc.id,
                    asset_exc,
                )

        doc.processing_status = ProcessingStatus.READY.value
        doc.chunk_count = len(chunks)
        doc.processed_at = datetime.now(UTC)
        doc.processing_error_code = None
        doc.processing_error_message = None

        self.db.commit()
        self.db.refresh(doc)

        logger.info(
            "Document processed and marked ready",
            extra={
                "extra_data": {
                    "document_id": str(doc.id),
                    "user_id": str(doc.user_id),
                    "chunk_count": doc.chunk_count,
                }
            },
        )
        return doc

    def _record_failure(
        self,
        *,
        document_id: uuid.UUID,
        error_code: str,
        user_message: str,
        exc: Exception,
    ) -> Document | None:
        self.db.rollback()
        logger.warning(
            "Document processing failed",
            extra={
                "extra_data": {
                    "document_id": str(document_id),
                    "error_code": error_code,
                    "exception_type": type(exc).__name__,
                }
            },
        )

        stmt = select(Document).where(Document.id == document_id).with_for_update()
        doc = self.db.scalar(stmt)
        if doc is None:
            return None

        # Ensure incomplete chunks and assets are removed on failure
        self.db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        self.db.execute(delete(DocumentAsset).where(DocumentAsset.document_id == doc.id))
        doc.processing_status = ProcessingStatus.FAILED.value
        doc.processing_error_code = error_code[:64]
        doc.processing_error_message = user_message[:500]
        doc.chunk_count = 0
        doc.processed_at = None

        self.db.commit()
        self.db.refresh(doc)
        return doc
