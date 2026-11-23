import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.modules.exams.models import ExamStatus, QuestionType


class ExamCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    document_ids: list[uuid.UUID] = Field(
        default_factory=list,
        max_length=10,
        description="List of ready, user-owned document IDs to generate questions from.",
    )
    analysis_id: uuid.UUID | None = Field(
        default=None,
        description="Optional completed document analysis ID to ground question generation.",
    )
    focus_topic: str | None = Field(
        default=None,
        max_length=300,
        description="Optional focal theme or chapter to emphasize in generated questions.",
    )
    question_count: int = Field(
        default=10,
        ge=5,
        le=25,
        description="Total number of questions to generate (5 to 25).",
    )
    question_types: list[QuestionType] | None = Field(
        default=None,
        description="Allowed question types. Defaults to balanced mix of all types.",
    )

    @field_validator("document_ids")
    @classmethod
    def validate_unique_documents(cls, v: list[uuid.UUID]) -> list[uuid.UUID]:
        seen = set()
        deduped: list[uuid.UUID] = []
        for doc_id in v:
            if doc_id not in seen:
                seen.add(doc_id)
                deduped.append(doc_id)
        return deduped

    @model_validator(mode="after")
    def validate_source_specification(self) -> "ExamCreateRequest":
        if self.analysis_id is None and not self.document_ids:
            raise ValueError(
                "At least one document ID must be provided when analysis_id is not specified."
            )
        return self


class ExamSourceDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    file_extension: str
    file_size_bytes: int


class ExamQuestionStudentRead(BaseModel):
    """
    Question representation delivered to students during an active exam.

    SECURITY INVARIANT:
    `correct_answer`, `explanation`, and `grading_rubric` are strictly omitted.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    question_index: int
    question_type: str
    prompt: str
    options: list[str] | None = None


class ExamAttemptStartResponse(BaseModel):
    attempt_id: uuid.UUID
    reviewer_id: uuid.UUID
    reviewer_title: str
    attempt_number: int = 1
    status: str
    started_at: datetime
    questions: list[ExamQuestionStudentRead]


class StudentAnswerItemSubmission(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question_id: uuid.UUID
    answer: str = Field(..., max_length=4000)


class ExamAttemptSubmitRequest(BaseModel):
    answers: list[StudentAnswerItemSubmission] = Field(
        ...,
        min_length=1,
        description="Student responses for exam questions.",
    )


class StudentAnswerResultRead(BaseModel):
    id: uuid.UUID
    question_id: uuid.UUID
    question_index: int
    question_type: str
    prompt: str
    options: list[str] | None
    student_answer: str
    correct_answer: str
    is_correct: bool | None
    score: float
    max_score: float = 1.0
    evaluator: str
    evaluator_type: str = ""
    explanation: str
    feedback: str | None
    source_reference: dict[str, Any] | None

    def model_post_init(self, __context: Any) -> None:
        if not self.evaluator_type:
            self.evaluator_type = self.evaluator


class ExamAttemptSummaryRead(BaseModel):
    id: uuid.UUID
    attempt_number: int
    status: str
    total_score: float | None = None
    max_score: float | None = None
    percentage: float | None = None
    started_at: datetime
    completed_at: datetime | None = None


class ExamResultRead(BaseModel):
    attempt_id: uuid.UUID
    reviewer_id: uuid.UUID
    reviewer_title: str
    attempt_number: int = 1
    status: str
    total_score: float
    max_score: float
    percentage: float
    feedback_summary: str | None
    started_at: datetime
    completed_at: datetime | None
    answers: list[StudentAnswerResultRead]


class ExamReviewerSummaryRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    analysis_id: uuid.UUID | None = None
    title: str
    focus_topic: str | None
    status: ExamStatus
    question_count: int
    question_types: list[str] = Field(default_factory=list)
    source_documents: list[ExamSourceDocumentRead]
    completed_attempts_count: int = 0
    attempts_count: int = 0
    best_percentage: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ExamReviewerDetailRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    analysis_id: uuid.UUID | None = None
    title: str
    focus_topic: str | None
    status: ExamStatus
    question_count: int
    question_types: list[str] = Field(default_factory=list)
    source_documents: list[ExamSourceDocumentRead]
    completed_attempts_count: int = 0
    attempts_count: int = 0
    best_percentage: float | None = None
    questions_available: bool = False
    latest_attempt: ExamAttemptSummaryRead | None = None
    attempts: list[ExamAttemptSummaryRead] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ExamReviewerListResponse(BaseModel):
    items: list[ExamReviewerSummaryRead]
    total: int
    limit: int
    offset: int


class ExamDeleteResponse(BaseModel):
    deleted: bool
