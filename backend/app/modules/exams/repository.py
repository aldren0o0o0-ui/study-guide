import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.exams.generator import GroundedQuestionData
from app.modules.exams.models import (
    AttemptStatus,
    ExamAttempt,
    ExamQuestion,
    ExamReviewer,
    ExamReviewerDocument,
    ExamStatus,
    StudentAnswer,
)


class ExamRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_reviewer(
        self,
        *,
        user_id: uuid.UUID,
        title: str,
        focus_topic: str | None,
        question_count: int,
        document_ids: Sequence[uuid.UUID],
        analysis_id: uuid.UUID | None = None,
    ) -> ExamReviewer:
        reviewer = ExamReviewer(
            user_id=user_id,
            analysis_id=analysis_id,
            title=title,
            focus_topic=focus_topic,
            question_count=question_count,
            status=ExamStatus.QUEUED.value,
        )
        self.db.add(reviewer)
        self.db.flush()

        for doc_id in document_ids:
            link = ExamReviewerDocument(
                reviewer_id=reviewer.id,
                document_id=doc_id,
            )
            self.db.add(link)

        self.db.flush()
        return reviewer

    def get_reviewer_for_user(
        self,
        reviewer_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ExamReviewer | None:
        stmt = (
            select(ExamReviewer)
            .options(
                selectinload(ExamReviewer.reviewer_documents).selectinload(
                    ExamReviewerDocument.document
                ),
                selectinload(ExamReviewer.questions),
                selectinload(ExamReviewer.attempts),
            )
            .where(
                ExamReviewer.id == reviewer_id,
                ExamReviewer.user_id == user_id,
            )
        )
        return self.db.scalar(stmt)

    def list_reviewers_for_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ExamReviewer], int]:
        count_stmt = select(func.count(ExamReviewer.id)).where(ExamReviewer.user_id == user_id)
        total = self.db.scalar(count_stmt) or 0

        stmt = (
            select(ExamReviewer)
            .options(
                selectinload(ExamReviewer.reviewer_documents).selectinload(
                    ExamReviewerDocument.document
                ),
                selectinload(ExamReviewer.attempts),
            )
            .where(ExamReviewer.user_id == user_id)
            .order_by(ExamReviewer.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list(self.db.scalars(stmt).all())
        return items, total

    def delete_reviewer(self, reviewer: ExamReviewer) -> None:
        self.db.delete(reviewer)

    def save_generated_questions(
        self,
        reviewer_id: uuid.UUID,
        questions: Sequence[GroundedQuestionData],
    ) -> list[ExamQuestion]:
        # Atomically clear any existing questions for this reviewer on regeneration retry
        self.db.execute(delete(ExamQuestion).where(ExamQuestion.reviewer_id == reviewer_id))
        self.db.flush()

        created: list[ExamQuestion] = []
        for q in questions:
            row = ExamQuestion(
                reviewer_id=reviewer_id,
                question_index=q.question_index,
                question_type=q.question_type,
                prompt=q.prompt,
                options=q.options,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                grading_rubric=q.grading_rubric,
                source_reference=q.source_reference,
            )
            self.db.add(row)
            created.append(row)
        self.db.flush()
        return created

    def create_attempt(
        self,
        *,
        reviewer_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ExamAttempt:
        stmt = select(func.coalesce(func.max(ExamAttempt.attempt_number), 0)).where(
            ExamAttempt.reviewer_id == reviewer_id,
            ExamAttempt.user_id == user_id,
        )
        max_num = self.db.scalar(stmt) or 0

        attempt = ExamAttempt(
            reviewer_id=reviewer_id,
            user_id=user_id,
            attempt_number=max_num + 1,
            status=AttemptStatus.IN_PROGRESS.value,
            started_at=datetime.now(UTC),
        )
        self.db.add(attempt)
        self.db.flush()
        return attempt

    def get_attempt_for_user(
        self,
        attempt_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ExamAttempt | None:
        stmt = (
            select(ExamAttempt)
            .options(
                selectinload(ExamAttempt.reviewer).selectinload(ExamReviewer.questions),
                selectinload(ExamAttempt.answers).selectinload(StudentAnswer.question),
            )
            .where(
                ExamAttempt.id == attempt_id,
                ExamAttempt.user_id == user_id,
            )
        )
        return self.db.scalar(stmt)

    def save_student_answer(
        self,
        *,
        attempt_id: uuid.UUID,
        question_id: uuid.UUID,
        student_answer: str,
        is_correct: bool | None,
        score: float,
        evaluator: str,
        feedback: str | None,
    ) -> StudentAnswer:
        ans = StudentAnswer(
            attempt_id=attempt_id,
            question_id=question_id,
            student_answer=student_answer,
            is_correct=is_correct,
            score=score,
            evaluator=evaluator,
            feedback=feedback,
        )
        self.db.add(ans)
        self.db.flush()
        return ans

    def complete_attempt(
        self,
        attempt: ExamAttempt,
        *,
        total_score: float,
        max_score: float,
        percentage: float,
        feedback_summary: str | None,
    ) -> ExamAttempt:
        attempt.status = AttemptStatus.COMPLETED.value
        attempt.total_score = total_score
        attempt.max_score = max_score
        attempt.percentage = percentage
        attempt.feedback_summary = feedback_summary
        attempt.completed_at = datetime.now(UTC)
        self.db.flush()
        return attempt
