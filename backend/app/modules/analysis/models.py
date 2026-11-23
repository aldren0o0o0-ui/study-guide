import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.modules.documents.models import Document


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentAnalysis(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document_analyses"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')",
            name="status_valid",
        ),
        Index("ix_document_analyses_user_id", "user_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    focus_topic: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
        default=None,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AnalysisStatus.QUEUED.value,
        server_default=AnalysisStatus.QUEUED.value,
    )
    result_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
    )
    error_message: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        default=None,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    analysis_documents: Mapped[list["AnalysisDocument"]] = relationship(
        "AnalysisDocument",
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="AnalysisDocument.position.asc()",
    )


class AnalysisDocument(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "analysis_documents"
    __table_args__ = (
        UniqueConstraint(
            "analysis_id",
            "document_id",
            name="uq_analysis_documents_analysis_doc",
        ),
        CheckConstraint("position >= 0", name="position_non_negative"),
        Index("ix_analysis_documents_analysis_id", "analysis_id"),
        Index("ix_analysis_documents_document_id", "document_id"),
    )

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_analyses.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    analysis: Mapped["DocumentAnalysis"] = relationship(
        "DocumentAnalysis",
        back_populates="analysis_documents",
    )
    document: Mapped["Document"] = relationship("Document")
