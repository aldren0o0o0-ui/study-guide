import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.modules.documents.models import Document, DocumentChunk


class VoiceSessionStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class VoiceEvaluationStatus(StrEnum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VoiceTurnStatus(StrEnum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


class VoiceSpeakerRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class VoicePracticeSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "voice_practice_sessions"
    __table_args__ = (
        CheckConstraint(
            "mode IN ('interview', 'communication', 'document_oral', 'neutral')",
            name="voice_session_mode_valid",
        ),
        CheckConstraint(
            "status IN ('pending', 'active', 'completed', 'interrupted', 'failed')",
            name="voice_session_status_valid",
        ),
        CheckConstraint(
            "evaluation_status IN ('not_started', 'pending', 'processing', 'completed', 'failed')",
            name="voice_session_eval_status_valid",
        ),
        CheckConstraint(
            "duration_seconds >= 0",
            name="voice_session_duration_non_negative",
        ),
        Index("ix_voice_practice_sessions_user_id", "user_id"),
        Index("ix_voice_practice_sessions_user_created", "user_id", "created_at"),
        Index("ix_voice_practice_sessions_status", "status"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=VoiceSessionStatus.PENDING.value,
        server_default=VoiceSessionStatus.PENDING.value,
    )
    evaluation_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=VoiceEvaluationStatus.NOT_STARTED.value,
        server_default=VoiceEvaluationStatus.NOT_STARTED.value,
    )
    config_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    termination_reason: Mapped[str | None] = mapped_column(
        String(64),
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
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    duration_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    turns: Mapped[list["VoicePracticeTurn"]] = relationship(
        "VoicePracticeTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="VoicePracticeTurn.turn_index.asc()",
    )
    session_documents: Mapped[list["VoicePracticeSessionDocument"]] = relationship(
        "VoicePracticeSessionDocument",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="VoicePracticeSessionDocument.position.asc()",
    )
    session_chunks: Mapped[list["VoicePracticeSessionChunk"]] = relationship(
        "VoicePracticeSessionChunk",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="VoicePracticeSessionChunk.position.asc()",
    )
    feedback: Mapped["VoicePracticeFeedback | None"] = relationship(
        "VoicePracticeFeedback",
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
    )


class VoicePracticeTurn(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "voice_practice_turns"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "turn_index",
            name="uq_voice_practice_turns_session_turn_idx",
        ),
        CheckConstraint(
            "turn_index >= 1",
            name="voice_turn_index_positive",
        ),
        CheckConstraint(
            "speaker_role IN ('user', 'assistant')",
            name="voice_turn_speaker_role_valid",
        ),
        CheckConstraint(
            "turn_status IN ('completed', 'interrupted')",
            name="voice_turn_status_valid",
        ),
        Index("ix_voice_practice_turns_session_id", "session_id"),
        Index("ix_voice_practice_turns_session_turn_idx", "session_id", "turn_index"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("voice_practice_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    speaker_role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    transcript_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    turn_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=VoiceTurnStatus.COMPLETED.value,
        server_default=VoiceTurnStatus.COMPLETED.value,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )

    session: Mapped["VoicePracticeSession"] = relationship(
        "VoicePracticeSession",
        back_populates="turns",
    )


class VoicePracticeFeedback(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "voice_practice_feedback"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            name="uq_voice_practice_feedback_session_id",
        ),
        CheckConstraint(
            "evaluation_status IN ('pending', 'processing', 'completed', 'failed')",
            name="voice_feedback_status_valid",
        ),
        CheckConstraint(
            "overall_score IS NULL OR (overall_score >= 0 AND overall_score <= 100)",
            name="voice_feedback_score_range",
        ),
        Index("ix_voice_practice_feedback_session_id", "session_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("voice_practice_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    evaluation_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=VoiceEvaluationStatus.PENDING.value,
        server_default=VoiceEvaluationStatus.PENDING.value,
    )
    overall_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        default=None,
    )
    dimension_scores_json: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )
    strengths_json: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )
    areas_for_improvement_json: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )
    recommendations_json: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )
    structured_feedback_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )
    model_name: Mapped[str | None] = mapped_column(
        String(128),
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

    session: Mapped["VoicePracticeSession"] = relationship(
        "VoicePracticeSession",
        back_populates="feedback",
    )


class VoicePracticeSessionDocument(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "voice_practice_session_documents"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "document_id",
            name="uq_voice_practice_session_docs_session_doc",
        ),
        CheckConstraint("position >= 0", name="voice_session_doc_pos_non_negative"),
        Index("ix_voice_practice_session_documents_session_id", "session_id"),
        Index("ix_voice_practice_session_documents_document_id", "document_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("voice_practice_sessions.id", ondelete="CASCADE"),
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped["VoicePracticeSession"] = relationship(
        "VoicePracticeSession",
        back_populates="session_documents",
    )
    document: Mapped["Document"] = relationship("Document")


class VoicePracticeSessionChunk(UUIDPrimaryKeyMixin, Base):
    """
    Preserves exact traceability to the original `Document` and `DocumentChunk` records
    retrieved for a Document-Based Oral Practice session without duplicating document text.
    """

    __tablename__ = "voice_practice_session_chunks"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "chunk_id",
            name="uq_voice_practice_session_chunks_session_chunk",
        ),
        UniqueConstraint(
            "session_id",
            "source_key",
            name="uq_voice_practice_session_chunks_source_key",
        ),
        CheckConstraint("position >= 0", name="voice_session_chunk_pos_non_negative"),
        Index("ix_voice_practice_session_chunks_session_id", "session_id"),
        Index("ix_voice_practice_session_chunks_document_id", "document_id"),
        Index("ix_voice_practice_session_chunks_chunk_id", "chunk_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("voice_practice_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_key: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    similarity_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    session: Mapped["VoicePracticeSession"] = relationship(
        "VoicePracticeSession",
        back_populates="session_chunks",
    )
    document: Mapped["Document"] = relationship("Document")
    chunk: Mapped["DocumentChunk"] = relationship("DocumentChunk")
