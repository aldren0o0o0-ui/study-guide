import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.database.vector import DEFAULT_EMBEDDING_DIMENSION, EmbeddingVector


class ProcessingStatus(StrEnum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("file_size_bytes > 0", name="file_size_positive"),
        CheckConstraint(
            "file_extension IN ('pdf', 'docx', 'txt')",
            name="file_extension_supported",
        ),
        CheckConstraint(
            "processing_status IN ('uploaded', 'queued', 'processing', 'ready', 'failed')",
            name="processing_status_valid",
        ),
        CheckConstraint("chunk_count >= 0", name="chunk_count_non_negative"),
        Index("ix_documents_user_id", "user_id"),
        Index("ix_documents_storage_key", "storage_key", unique=True),
        Index("ix_documents_user_checksum", "user_id", "checksum_sha256"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    file_extension: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    checksum_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    processing_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ProcessingStatus.UPLOADED.value,
        server_default=ProcessingStatus.UPLOADED.value,
    )
    processing_error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
    )
    processing_error_message: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        default=None,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index.asc()",
    )
    assets: Mapped[list["DocumentAsset"]] = relationship(
        "DocumentAsset",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentAsset.order_index.asc()",
    )


class DocumentChunk(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_doc_chunk_idx"),
        CheckConstraint("chunk_index >= 0", name="chunk_index_non_negative"),
        CheckConstraint("char_count > 0", name="char_count_positive"),
        CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) "
            "OR (page_start >= 1 AND page_end >= page_start)",
            name="page_range_valid",
        ),
        Index("ix_document_chunks_document_id", "document_id"),
        Index("ix_document_chunks_user_id", "user_id"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    char_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    page_start: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )
    page_end: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )
    section_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        default=None,
    )
    embedding: Mapped[list[float]] = mapped_column(
        EmbeddingVector(DEFAULT_EMBEDDING_DIMENSION),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )


class DocumentAsset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Extracted visual asset (figure, diagram, photo, chart) preserved from an uploaded document.
    """

    __tablename__ = "document_assets"
    __table_args__ = (
        CheckConstraint(
            "file_size_bytes > 0",
            name="asset_file_size_positive",
        ),
        CheckConstraint(
            "order_index >= 0",
            name="asset_order_index_non_negative",
        ),
        Index("ix_document_assets_document_id", "document_id"),
        Index("ix_document_assets_user_id", "user_id"),
        Index("ix_document_assets_storage_key", "storage_key", unique=True),
        Index("ix_document_assets_doc_checksum", "document_id", "checksum_sha256"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    asset_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="image",
        server_default="image",
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    checksum_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )
    section_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        default=None,
    )
    caption: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        default=None,
    )
    order_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )
    height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="assets",
    )
