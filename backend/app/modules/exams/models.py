import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.modules.documents.models import Document

if TYPE_CHECKING:
    from app.modules.analysis.models import DocumentAnalysis


class ExamStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


class AttemptStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class EvaluatorType(StrEnum):
    DETERMINISTIC = "deterministic"
    AI_ASSISTED = "ai_assisted"


class ExamReviewer(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Represents an AI-generated exam reviewer synthesized from one or more
    processed, user-owned learning documents.
    """

    __tablename__ = "exam_reviewers"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    focus_topic: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ExamStatus.QUEUED.value,
    )

    question_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    analysis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_analyses.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        default=None,
    )

    # Relationships
    analysis: Mapped["DocumentAnalysis | None"] = relationship(
        "DocumentAnalysis",
        lazy="selectin",
    )

    reviewer_documents: Mapped[list["ExamReviewerDocument"]] = relationship(
        "ExamReviewerDocument",
        back_populates="reviewer",
        cascade="all, delete-orphan",
        order_by="ExamReviewerDocument.id",
    )

    questions: Mapped[list["ExamQuestion"]] = relationship(
        "ExamQuestion",
        back_populates="reviewer",
        cascade="all, delete-orphan",
        order_by="ExamQuestion.question_index",
    )

    attempts: Mapped[list["ExamAttempt"]] = relationship(
        "ExamAttempt",
        back_populates="reviewer",
        cascade="all, delete-orphan",
        order_by="ExamAttempt.started_at.desc()",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed')",
            name="chk_exam_reviewer_status",
        ),
        Index("idx_exam_reviewer_user_status", "user_id", "status"),
        Index("idx_exam_reviewer_user_created", "user_id", "created_at"),
        Index("idx_exam_reviewer_analysis_id", "analysis_id"),
    )


class ExamReviewerDocument(Base, UUIDPrimaryKeyMixin):
    """
    Join table associating an ExamReviewer with one or more source Documents.
    """

    __tablename__ = "exam_reviewer_documents"

    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exam_reviewers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Relationships
    reviewer: Mapped["ExamReviewer"] = relationship(
        "ExamReviewer",
        back_populates="reviewer_documents",
    )

    document: Mapped["Document"] = relationship(
        "Document",
        lazy="joined",
    )

    __table_args__ = (
        UniqueConstraint("reviewer_id", "document_id", name="uq_exam_reviewer_document"),
    )


class ExamQuestion(Base, UUIDPrimaryKeyMixin):
    """
    An individual question belonging to an ExamReviewer.

    SECURITY INVARIANT:
    `correct_answer`, `explanation`, and `grading_rubric` must NEVER be exposed
    to the student before or during an active exam attempt.
    """

    __tablename__ = "exam_questions"

    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exam_reviewers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    question_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    question_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    options: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    correct_answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    grading_rubric: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_reference: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Relationships
    reviewer: Mapped["ExamReviewer"] = relationship(
        "ExamReviewer",
        back_populates="questions",
    )

    student_answers: Mapped[list["StudentAnswer"]] = relationship(
        "StudentAnswer",
        back_populates="question",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "question_type IN ('multiple_choice', 'true_false', 'short_answer')",
            name="chk_exam_question_type",
        ),
        UniqueConstraint("reviewer_id", "question_index", name="uq_exam_question_index"),
        Index("idx_exam_question_reviewer_type", "reviewer_id", "question_type"),
    )


class ExamAttempt(Base, UUIDPrimaryKeyMixin):
    """
    A student's session taking an ExamReviewer. Supports retakes and attempt history.
    """

    __tablename__ = "exam_attempts"

    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exam_reviewers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AttemptStatus.IN_PROGRESS.value,
    )

    total_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    max_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    feedback_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    reviewer: Mapped["ExamReviewer"] = relationship(
        "ExamReviewer",
        back_populates="attempts",
    )

    answers: Mapped[list["StudentAnswer"]] = relationship(
        "StudentAnswer",
        back_populates="attempt",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('in_progress', 'completed')",
            name="chk_exam_attempt_status",
        ),
        Index("idx_exam_attempt_user_reviewer", "user_id", "reviewer_id"),
        Index("idx_exam_attempt_reviewer_number", "reviewer_id", "attempt_number"),
        Index("idx_exam_attempt_user_started", "user_id", "started_at"),
    )


class StudentAnswer(Base, UUIDPrimaryKeyMixin):
    """
    A student's recorded answer and grade for a single question in an attempt.
    """

    __tablename__ = "student_answers"

    attempt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exam_attempts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exam_questions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    student_answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    is_correct: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    evaluator: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=EvaluatorType.DETERMINISTIC.value,
    )

    feedback: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Relationships
    attempt: Mapped["ExamAttempt"] = relationship(
        "ExamAttempt",
        back_populates="answers",
    )

    question: Mapped["ExamQuestion"] = relationship(
        "ExamQuestion",
        back_populates="student_answers",
        lazy="joined",
    )

    __table_args__ = (
        UniqueConstraint("attempt_id", "question_id", name="uq_student_attempt_question"),
        CheckConstraint(
            "evaluator IN ('deterministic', 'ai_assisted')",
            name="chk_student_answer_evaluator",
        ),
    )
