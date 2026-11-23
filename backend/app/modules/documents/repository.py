import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.documents.models import Document, ProcessingStatus


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_document_for_user(self, document_id: uuid.UUID, user_id: uuid.UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
        return self.db.scalar(stmt)

    def list_documents_for_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        base_filter = [Document.user_id == user_id]

        count_stmt = select(func.count(Document.id)).where(*base_filter)
        total = int(self.db.scalar(count_stmt) or 0)

        items_stmt = (
            select(Document)
            .where(*base_filter)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list(self.db.scalars(items_stmt).all())
        return items, total

    def create_document(
        self,
        *,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        original_filename: str,
        storage_key: str,
        file_extension: str,
        mime_type: str,
        file_size_bytes: int,
        checksum_sha256: str,
    ) -> Document:
        doc = Document(
            id=document_id,
            user_id=user_id,
            original_filename=original_filename,
            storage_key=storage_key,
            file_extension=file_extension,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            checksum_sha256=checksum_sha256,
            processing_status=ProcessingStatus.UPLOADED.value,
        )
        self.db.add(doc)
        self.db.flush()
        return doc

    def delete_document(self, document: Document) -> None:
        self.db.delete(document)
        self.db.flush()
