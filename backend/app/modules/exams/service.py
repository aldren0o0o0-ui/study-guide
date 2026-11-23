import logging
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AppException,
    BadRequestException,
    ConflictException,
    NotFoundException,
    ServiceUnavailableException,
)
from app.modules.analysis.models import AnalysisStatus, DocumentAnalysis
from app.modules.analysis.schemas import StructuredAnalysisResult
from app.modules.analysis.service import select_chunks_within_budget
from app.modules.documents.models import Document, DocumentChunk, ProcessingStatus
from app.modules.exams.exceptions import ExamError, ExamPermanentError
from app.modules.exams.generator import (
    ExamGenerator,
    GroundedQuestionData,
    create_exam_generator,
)
from app.modules.exams.grader import (
    SubjectiveGrader,
    create_subjective_grader,
    grade_objective_answer,
)
from app.modules.exams.models import (
    AttemptStatus,
    EvaluatorType,
    ExamAttempt,
    ExamQuestion,
    ExamReviewer,
    ExamReviewerDocument,
    ExamStatus,
    QuestionType,
)
from app.modules.exams.repository import ExamRepository
from app.modules.exams.schemas import (
    ExamAttemptStartResponse,
    ExamAttemptSubmitRequest,
    ExamAttemptSummaryRead,
    ExamCreateRequest,
    ExamDeleteResponse,
    ExamQuestionStudentRead,
    ExamResultRead,
    ExamReviewerDetailRead,
    ExamReviewerListResponse,
    ExamReviewerSummaryRead,
    ExamSourceDocumentRead,
    StudentAnswerResultRead,
)
from app.modules.processing.exceptions import DocumentProcessingError
from app.modules.processing.retrieval import RetrievalService, RetrievedChunk

logger = logging.getLogger("app.modules.exams.service")


class ExamQueueDispatcher(Protocol):
    def enqueue_exam(self, reviewer_id: uuid.UUID) -> str: ...


def _build_initial_exam_title(
    documents: Sequence[Document],
    focus_topic: str | None,
) -> str:
    if focus_topic:
        return f"Exam: {focus_topic}"[:255]
    if len(documents) == 1:
        return f"Exam on {documents[0].original_filename}"[:255]
    return f"Exam on {documents[0].original_filename} + {len(documents) - 1} more"[:255]


def _to_source_docs_read(reviewer: ExamReviewer) -> list[ExamSourceDocumentRead]:
    return [
        ExamSourceDocumentRead.model_validate(link.document)
        for link in reviewer.reviewer_documents
        if link.document is not None
    ]


def _compute_attempt_stats(reviewer: ExamReviewer) -> tuple[int, float | None]:
    completed = [
        att.percentage
        for att in reviewer.attempts
        if att.status == AttemptStatus.COMPLETED.value and att.percentage is not None
    ]
    if not completed:
        return 0, None
    best_pct = max(completed)
    return len(completed), round(best_pct, 1)


def _extract_question_types(reviewer: ExamReviewer) -> list[str]:
    types_set = {q.question_type for q in reviewer.questions}
    return sorted(types_set)


def _to_summary_read(reviewer: ExamReviewer) -> ExamReviewerSummaryRead:
    count, best = _compute_attempt_stats(reviewer)
    q_types = _extract_question_types(reviewer)
    return ExamReviewerSummaryRead(
        id=reviewer.id,
        user_id=reviewer.user_id,
        analysis_id=reviewer.analysis_id,
        title=reviewer.title,
        focus_topic=reviewer.focus_topic,
        status=ExamStatus(reviewer.status),
        question_count=reviewer.question_count,
        question_types=q_types,
        source_documents=_to_source_docs_read(reviewer),
        completed_attempts_count=count,
        attempts_count=count,
        best_percentage=best,
        error_code=reviewer.error_code,
        error_message=reviewer.error_message,
        created_at=reviewer.created_at,
        updated_at=reviewer.updated_at,
    )


def _to_detail_read(reviewer: ExamReviewer) -> ExamReviewerDetailRead:
    count, best = _compute_attempt_stats(reviewer)
    q_types = _extract_question_types(reviewer)
    attempts_summaries = [
        ExamAttemptSummaryRead(
            id=att.id,
            attempt_number=att.attempt_number,
            status=att.status,
            total_score=att.total_score,
            max_score=att.max_score,
            percentage=att.percentage,
            started_at=att.started_at,
            completed_at=att.completed_at,
        )
        for att in sorted(reviewer.attempts, key=lambda a: a.started_at, reverse=True)
    ]
    latest = attempts_summaries[0] if attempts_summaries else None
    return ExamReviewerDetailRead(
        id=reviewer.id,
        user_id=reviewer.user_id,
        analysis_id=reviewer.analysis_id,
        title=reviewer.title,
        focus_topic=reviewer.focus_topic,
        status=ExamStatus(reviewer.status),
        question_count=reviewer.question_count,
        question_types=q_types,
        source_documents=_to_source_docs_read(reviewer),
        completed_attempts_count=count,
        attempts_count=count,
        best_percentage=best,
        questions_available=reviewer.status == ExamStatus.COMPLETED.value,
        latest_attempt=latest,
        attempts=attempts_summaries,
        error_code=reviewer.error_code,
        error_message=reviewer.error_message,
        created_at=reviewer.created_at,
        updated_at=reviewer.updated_at,
    )


class ExamExecutionService:
    """
    Background worker pipeline that retrieves relevant context from user documents,
    prompts Gemini for structured questions, and persists them.
    """

    def __init__(
        self,
        db: Session,
        *,
        retrieval_service: RetrievalService | None = None,
        generator: ExamGenerator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.retrieval = retrieval_service or RetrievalService(db, settings=self.settings)
        self.generator = generator or create_exam_generator(self.settings)
        self.repo = ExamRepository(db)

    def execute_reviewer_generation(self, reviewer_id: uuid.UUID) -> ExamReviewer | None:
        stmt = (
            select(ExamReviewer)
            .options(
                selectinload(ExamReviewer.reviewer_documents).selectinload(
                    ExamReviewerDocument.document
                )
            )
            .where(ExamReviewer.id == reviewer_id)
            .with_for_update()
        )
        reviewer = self.db.scalar(stmt)
        if reviewer is None:
            self.db.rollback()
            logger.warning(
                "Exam reviewer not found when starting generation job",
                extra={"extra_data": {"reviewer_id": str(reviewer_id)}},
            )
            return None

        if reviewer.status == ExamStatus.PROCESSING.value:
            self.db.commit()
            logger.info(
                "Skipping duplicate exam generation; reviewer already processing",
                extra={"extra_data": {"reviewer_id": str(reviewer_id)}},
            )
            return reviewer

        reviewer.status = ExamStatus.PROCESSING.value
        reviewer.error_code = None
        reviewer.error_message = None
        self.db.commit()

        try:
            ready_docs = [
                link.document
                for link in reviewer.reviewer_documents
                if link.document is not None
                and link.document.user_id == reviewer.user_id
                and link.document.processing_status == ProcessingStatus.READY.value
            ]

            if not ready_docs:
                raise ExamPermanentError(
                    "No ready source documents remain associated with this reviewer",
                    code="SOURCE_DOCUMENTS_UNAVAILABLE",
                    user_message=(
                        "The source documents for this reviewer are no longer available or ready."
                    ),
                )

            analysis_result: StructuredAnalysisResult | None = None
            if reviewer.analysis_id:
                stmt_analysis = (
                    select(DocumentAnalysis)
                    .where(
                        DocumentAnalysis.id == reviewer.analysis_id,
                        DocumentAnalysis.user_id == reviewer.user_id,
                    )
                )
                analysis = self.db.scalar(stmt_analysis)
                if (
                    analysis is None
                    or analysis.status != AnalysisStatus.COMPLETED.value
                    or not analysis.result_json
                ):
                    raise ExamPermanentError(
                        "Associated document analysis is missing, unready, or invalid",
                        code="ANALYSIS_UNAVAILABLE",
                        user_message=(
                            "The document analysis associated with this quiz is no longer "
                            "available or ready."
                        ),
                    )
                analysis_result = StructuredAnalysisResult.model_validate(analysis.result_json)

            if analysis_result and analysis_result.source_references:
                # Load the verified source chunks that grounded the analysis
                chunk_ids = [ref.chunk_id for ref in analysis_result.source_references]
                raw_chunks = self.db.scalars(
                    select(DocumentChunk).where(
                        DocumentChunk.id.in_(chunk_ids),
                        DocumentChunk.user_id == reviewer.user_id,
                    )
                ).all()
                chunk_map = {c.id: c for c in raw_chunks}
                selected_chunks: list[RetrievedChunk] = []
                for ref in analysis_result.source_references:
                    c = chunk_map.get(ref.chunk_id)
                    if c is not None:
                        selected_chunks.append(
                            RetrievedChunk(
                                chunk_id=c.id,
                                document_id=c.document_id,
                                original_filename=ref.original_filename,
                                chunk_index=c.chunk_index,
                                page_start=c.page_start,
                                page_end=c.page_end,
                                page_number=c.page_start,
                                section_title=c.section_title,
                                text=c.content,
                                distance=0.0,
                                similarity_score=1.0,
                            )
                        )
                if not selected_chunks:
                    selected_chunks = self._retrieve_and_budget_chunks(
                        user_id=reviewer.user_id,
                        documents=ready_docs,
                        focus_topic=reviewer.focus_topic,
                    )
            else:
                selected_chunks = self._retrieve_and_budget_chunks(
                    user_id=reviewer.user_id,
                    documents=ready_docs,
                    focus_topic=reviewer.focus_topic,
                )

            generated_result = self.generator.generate_exam(
                chunks=selected_chunks,
                question_count=reviewer.question_count,
                focus_topic=reviewer.focus_topic,
                analysis_result=analysis_result,
            )

            return self._persist_completed_reviewer(
                reviewer_id=reviewer_id,
                title=generated_result.title,
                questions=generated_result.questions,
            )
        except ExamError as exc:
            return self._record_failure(
                reviewer_id=reviewer_id,
                error_code=exc.code,
                user_message=exc.user_message,
                exc=exc,
            )
        except DocumentProcessingError as exc:
            return self._record_failure(
                reviewer_id=reviewer_id,
                error_code=exc.code,
                user_message=exc.user_message,
                exc=exc,
            )
        except AppException as exc:
            return self._record_failure(
                reviewer_id=reviewer_id,
                error_code=exc.code,
                user_message=exc.message,
                exc=exc,
            )
        except Exception as exc:
            return self._record_failure(
                reviewer_id=reviewer_id,
                error_code="INTERNAL_EXAM_ERROR",
                user_message=(
                    "An unexpected error occurred while generating this exam. Please try again."
                ),
                exc=exc,
            )

    def _retrieve_and_budget_chunks(
        self,
        *,
        user_id: uuid.UUID,
        documents: Sequence[Document],
        focus_topic: str | None,
    ) -> list[RetrievedChunk]:
        top_k = max(2, min(50, self.settings.ANALYSIS_TOP_K_CHUNKS))
        per_doc_k = max(2, min(50, top_k // max(1, len(documents))))

        chunks_by_doc: list[tuple[uuid.UUID, Sequence[RetrievedChunk]]] = []

        if focus_topic:
            for doc in documents:
                doc_hits = self.retrieval.retrieve_relevant_chunks(
                    user_id=user_id,
                    query=focus_topic,
                    document_ids=[doc.id],
                    limit=per_doc_k,
                )
                chunks_by_doc.append((doc.id, doc_hits))
        else:
            for doc in documents:
                overview_query = (
                    f"Core topics, concepts, definitions, and questions for {doc.original_filename}"
                )
                doc_hits = self.retrieval.retrieve_relevant_chunks(
                    user_id=user_id,
                    query=overview_query,
                    document_ids=[doc.id],
                    limit=per_doc_k,
                )
                chunks_by_doc.append((doc.id, doc_hits))

        selected_chunks, _ = select_chunks_within_budget(
            chunks_by_doc,
            max_context_chars=self.settings.ANALYSIS_MAX_CONTEXT_CHARS,
        )

        if not selected_chunks:
            raise ExamPermanentError(
                "No readable chunks could be retrieved from the selected documents",
                code="NO_SOURCE_CHUNKS_AVAILABLE",
                user_message="No readable content chunks were found in the selected documents.",
            )

        return selected_chunks

    def _persist_completed_reviewer(
        self,
        *,
        reviewer_id: uuid.UUID,
        title: str,
        questions: Sequence[GroundedQuestionData],
    ) -> ExamReviewer | None:
        reviewer = self.db.scalar(
            select(ExamReviewer).where(ExamReviewer.id == reviewer_id).with_for_update()
        )
        if reviewer is None:
            self.db.rollback()
            return None

        reviewer.title = title[:255]
        reviewer.question_count = len(questions)
        reviewer.status = ExamStatus.COMPLETED.value
        reviewer.error_code = None
        reviewer.error_message = None

        self.repo.save_generated_questions(reviewer_id, questions)
        self.db.commit()
        self.db.refresh(reviewer)
        return reviewer

    def _record_failure(
        self,
        *,
        reviewer_id: uuid.UUID,
        error_code: str,
        user_message: str,
        exc: Exception,
    ) -> ExamReviewer | None:
        self.db.rollback()
        logger.warning(
            "Exam reviewer generation failed",
            extra={
                "extra_data": {
                    "reviewer_id": str(reviewer_id),
                    "error_code": error_code,
                    "exception_type": type(exc).__name__,
                }
            },
        )
        reviewer = self.db.scalar(
            select(ExamReviewer).where(ExamReviewer.id == reviewer_id).with_for_update()
        )
        if reviewer is None:
            return None

        reviewer.status = ExamStatus.FAILED.value
        reviewer.error_code = error_code[:64]
        reviewer.error_message = user_message[:500]
        self.db.commit()
        self.db.refresh(reviewer)
        return reviewer


class ExamReviewerService:
    def __init__(
        self,
        db: Session,
        *,
        exam_queue: ExamQueueDispatcher,
        subjective_grader: SubjectiveGrader | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.repo = ExamRepository(db)
        self.exam_queue = exam_queue
        self.settings = settings or get_settings()
        self.grader = subjective_grader or create_subjective_grader(self.settings)

    def create_reviewer(
        self,
        user_id: uuid.UUID,
        payload: ExamCreateRequest,
    ) -> ExamReviewerDetailRead:
        analysis: DocumentAnalysis | None = None
        if payload.analysis_id:
            stmt = (
                select(DocumentAnalysis)
                .options(selectinload(DocumentAnalysis.analysis_documents))
                .where(
                    DocumentAnalysis.id == payload.analysis_id,
                    DocumentAnalysis.user_id == user_id,
                )
            )
            analysis = self.db.scalar(stmt)
            if analysis is None:
                raise NotFoundException(
                    "Document analysis not found or inaccessible",
                    code="ANALYSIS_NOT_FOUND",
                )
            if analysis.status != AnalysisStatus.COMPLETED.value:
                raise BadRequestException(
                    f"Document analysis is not completed (current status: {analysis.status})",
                    code="ANALYSIS_NOT_COMPLETED",
                )
            if not analysis.result_json:
                raise BadRequestException(
                    "Document analysis contains no generated pedagogical content",
                    code="ANALYSIS_RESULT_EMPTY",
                )

            analysis_doc_ids = [link.document_id for link in analysis.analysis_documents]
            if not analysis_doc_ids:
                raise BadRequestException(
                    "Analysis has no associated source documents",
                    code="ANALYSIS_DOCUMENTS_EMPTY",
                )

            if payload.document_ids:
                analysis_doc_set = set(analysis_doc_ids)
                if not set(payload.document_ids).issubset(analysis_doc_set):
                    raise BadRequestException(
                        "Specified document_ids do not match the source documents "
                        "of the specified analysis",
                        code="DOCUMENT_ANALYSIS_MISMATCH",
                    )
                target_doc_ids = payload.document_ids
            else:
                target_doc_ids = analysis_doc_ids
        else:
            target_doc_ids = payload.document_ids
            if not target_doc_ids:
                raise BadRequestException(
                    "At least one document ID must be provided",
                    code="NO_DOCUMENTS_SELECTED",
                )

        if len(target_doc_ids) > self.settings.ANALYSIS_MAX_DOCUMENTS:
            raise BadRequestException(
                f"Cannot select more than {self.settings.ANALYSIS_MAX_DOCUMENTS} documents",
                code="TOO_MANY_DOCUMENTS_SELECTED",
            )

        ordered_docs = self._validate_and_load_ready_documents(
            user_id=user_id,
            document_ids=target_doc_ids,
        )

        if analysis and not payload.focus_topic:
            initial_title = f"Quiz: {analysis.title}"[:255]
        else:
            initial_title = _build_initial_exam_title(ordered_docs, payload.focus_topic)

        reviewer = self.repo.create_reviewer(
            user_id=user_id,
            title=initial_title,
            focus_topic=payload.focus_topic,
            question_count=payload.question_count,
            document_ids=[doc.id for doc in ordered_docs],
            analysis_id=analysis.id if analysis else None,
        )
        self.db.commit()

        try:
            self.exam_queue.enqueue_exam(reviewer.id)
        except Exception as exc:
            logger.warning(
                "Failed to enqueue exam reviewer job",
                extra={"extra_data": {"reviewer_id": str(reviewer.id), "error": str(exc)}},
            )
            reviewer.status = ExamStatus.FAILED.value
            reviewer.error_code = "QUEUE_UNAVAILABLE"
            reviewer.error_message = (
                "Background exam queue is temporarily unavailable. Please tap Retry."
            )
            self.db.commit()

        refreshed = self.repo.get_reviewer_for_user(reviewer.id, user_id)
        assert refreshed is not None
        return _to_detail_read(refreshed)

    def _validate_and_load_ready_documents(
        self,
        *,
        user_id: uuid.UUID,
        document_ids: Sequence[uuid.UUID],
    ) -> list[Document]:
        stmt = select(Document).where(
            Document.id.in_(document_ids),
            Document.user_id == user_id,
        )
        found_docs = {doc.id: doc for doc in self.db.scalars(stmt).all()}

        ordered_docs: list[Document] = []
        for doc_id in document_ids:
            doc = found_docs.get(doc_id)
            if doc is None:
                raise NotFoundException(
                    "One or more selected documents were not found",
                    code="DOCUMENT_NOT_FOUND",
                )
            if doc.processing_status != ProcessingStatus.READY.value:
                raise BadRequestException(
                    f"Document '{doc.original_filename}' is not ready for exam generation "
                    f"(current status: {doc.processing_status})",
                    code="DOCUMENT_NOT_READY",
                )
            ordered_docs.append(doc)

        return ordered_docs

    def list_reviewers(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ExamReviewerListResponse:
        items, total = self.repo.list_reviewers_for_user(user_id, limit=limit, offset=offset)
        return ExamReviewerListResponse(
            items=[_to_summary_read(r) for r in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_reviewer(
        self,
        user_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> ExamReviewerDetailRead:
        reviewer = self.repo.get_reviewer_for_user(reviewer_id, user_id)
        if reviewer is None:
            raise NotFoundException("Exam reviewer not found", code="EXAM_NOT_FOUND")
        return _to_detail_read(reviewer)

    def retry_reviewer(
        self,
        user_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> ExamReviewerDetailRead:
        reviewer = self.repo.get_reviewer_for_user(reviewer_id, user_id)
        if reviewer is None:
            raise NotFoundException("Exam reviewer not found", code="EXAM_NOT_FOUND")

        if reviewer.status == ExamStatus.PROCESSING.value:
            raise ConflictException(
                "Exam is currently being generated",
                code="EXAM_ALREADY_PROCESSING",
            )
        if reviewer.status == ExamStatus.COMPLETED.value:
            raise ConflictException(
                "Exam has already been generated successfully",
                code="EXAM_ALREADY_COMPLETED",
            )

        reviewer.status = ExamStatus.QUEUED.value
        reviewer.error_code = None
        reviewer.error_message = None
        self.db.commit()

        try:
            self.exam_queue.enqueue_exam(reviewer.id)
        except Exception as exc:
            reviewer.status = ExamStatus.FAILED.value
            reviewer.error_code = "QUEUE_UNAVAILABLE"
            reviewer.error_message = "Background exam queue is temporarily unavailable."
            self.db.commit()
            raise ServiceUnavailableException(
                "Background exam queue is temporarily unavailable.",
                code="QUEUE_UNAVAILABLE",
            ) from exc

        refreshed = self.repo.get_reviewer_for_user(reviewer.id, user_id)
        assert refreshed is not None
        return _to_detail_read(refreshed)

    def delete_reviewer(
        self,
        user_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> ExamDeleteResponse:
        reviewer = self.repo.get_reviewer_for_user(reviewer_id, user_id)
        if reviewer is None:
            raise NotFoundException("Exam reviewer not found", code="EXAM_NOT_FOUND")

        self.repo.delete_reviewer(reviewer)
        self.db.commit()
        return ExamDeleteResponse(deleted=True)

    def start_attempt(
        self,
        user_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> ExamAttemptStartResponse:
        """
        Starts a new exam session for the student.

        ANTI-CHEAT INVARIANT:
        Returns `ExamQuestionStudentRead` with zero correct answers or explanations.
        """
        reviewer = self.repo.get_reviewer_for_user(reviewer_id, user_id)
        if reviewer is None:
            raise NotFoundException("Exam reviewer not found", code="EXAM_NOT_FOUND")

        if reviewer.status != ExamStatus.COMPLETED.value:
            raise BadRequestException(
                f"Cannot start exam while reviewer is {reviewer.status}",
                code="EXAM_NOT_READY",
            )

        if not reviewer.questions:
            raise BadRequestException(
                "This exam reviewer has no questions generated.",
                code="NO_QUESTIONS_AVAILABLE",
            )

        attempt = self.repo.create_attempt(reviewer_id=reviewer_id, user_id=user_id)
        self.db.commit()

        # Build student-safe questions (stripping answer keys & explanations)
        student_questions = [
            ExamQuestionStudentRead(
                id=q.id,
                question_index=q.question_index,
                question_type=q.question_type,
                prompt=q.prompt,
                options=q.options,
            )
            for q in reviewer.questions
        ]

        return ExamAttemptStartResponse(
            attempt_id=attempt.id,
            reviewer_id=reviewer.id,
            reviewer_title=reviewer.title,
            attempt_number=attempt.attempt_number,
            status=attempt.status,
            started_at=attempt.started_at,
            questions=student_questions,
        )

    def get_attempt(
        self,
        user_id: uuid.UUID,
        attempt_id: uuid.UUID,
    ) -> ExamAttemptStartResponse:
        """
        Retrieves active in-progress attempt masked questions for resuming an exam session.
        ANTI-CHEAT INVARIANT: Strictly omits answer keys and explanations.
        """
        attempt = self.repo.get_attempt_for_user(attempt_id, user_id)
        if attempt is None:
            raise NotFoundException("Exam attempt not found", code="ATTEMPT_NOT_FOUND")

        reviewer = attempt.reviewer
        student_questions = [
            ExamQuestionStudentRead(
                id=q.id,
                question_index=q.question_index,
                question_type=q.question_type,
                prompt=q.prompt,
                options=q.options,
            )
            for q in reviewer.questions
        ]
        return ExamAttemptStartResponse(
            attempt_id=attempt.id,
            reviewer_id=reviewer.id,
            reviewer_title=reviewer.title,
            attempt_number=attempt.attempt_number,
            status=attempt.status,
            started_at=attempt.started_at,
            questions=student_questions,
        )

    def submit_attempt(
        self,
        user_id: uuid.UUID,
        attempt_id: uuid.UUID,
        payload: ExamAttemptSubmitRequest,
    ) -> ExamResultRead:
        """
        Grades objective and subjective questions authoritatively on the backend.
        Optimized concurrency: Subjective AI calls occur outside open row locks.
        Preserves answer integrity: Zero fabricated fallback scores on AI failure.
        """
        # Step 1: Pre-validation without holding row lock
        attempt_pre = self.repo.get_attempt_for_user(attempt_id, user_id)
        if attempt_pre is None:
            raise NotFoundException("Exam attempt not found", code="ATTEMPT_NOT_FOUND")
        if attempt_pre.status == AttemptStatus.COMPLETED.value:
            raise ConflictException(
                "This exam attempt has already been submitted and graded.",
                code="ATTEMPT_ALREADY_COMPLETED",
            )

        reviewer = attempt_pre.reviewer
        answers_map: dict[uuid.UUID, str] = {
            item.question_id: item.answer for item in payload.answers
        }

        # Step 2: Evaluate questions in-memory / external API
        # Collect subjective questions for parallel evaluation
        subjective_tasks: list[tuple[ExamQuestion, str, str | None]] = []
        for q in reviewer.questions:
            student_ans_str = answers_map.get(q.id, "").strip()
            if q.question_type not in (
                QuestionType.MULTIPLE_CHOICE.value,
                QuestionType.TRUE_FALSE.value,
            ):
                source_excerpt = None
                if q.source_reference and isinstance(q.source_reference, dict):
                    source_excerpt = q.source_reference.get("excerpt")
                subjective_tasks.append((q, student_ans_str, source_excerpt))

        subjective_results: dict[uuid.UUID, tuple[float, bool, str | None]] = {}
        if subjective_tasks:
            def _eval_one(
                item: tuple[ExamQuestion, str, str | None],
            ) -> tuple[uuid.UUID, float, bool, str | None]:
                q_item, ans_str, excerpt = item
                sub_res = self.grader.grade_short_answer(
                    prompt=q_item.prompt,
                    correct_answer=q_item.correct_answer,
                    grading_rubric=q_item.grading_rubric,
                    source_excerpt=excerpt,
                    student_answer=ans_str,
                )
                return q_item.id, sub_res.score, sub_res.is_correct, sub_res.feedback

            with ThreadPoolExecutor(max_workers=min(5, len(subjective_tasks))) as executor:
                futures = {executor.submit(_eval_one, task): task for task in subjective_tasks}
                for fut in as_completed(futures):
                    task_q, _, _ = futures[fut]
                    try:
                        q_id, sc, is_c, fb = fut.result()
                        subjective_results[q_id] = (sc, is_c, fb)
                    except Exception as exc:
                        logger.warning(
                            "AI evaluation service failed during attempt submission",
                            extra={
                                "extra_data": {
                                    "question_id": str(task_q.id),
                                    "error": str(exc),
                                }
                            },
                        )
                        raise ServiceUnavailableException(
                            "AI evaluation service is temporarily unavailable. "
                            "Please tap Submit Exam again in a moment to finalize grading.",
                            code="AI_EVALUATION_UNAVAILABLE",
                        ) from exc

        # Assemble graded items in original question order
        graded_items: list[tuple[uuid.UUID, str, bool, float, str, str | None]] = []
        total_score = 0.0
        max_score = float(len(reviewer.questions))

        for q in reviewer.questions:
            student_ans_str = answers_map.get(q.id, "").strip()
            is_correct: bool
            score: float
            feedback_text: str | None

            if q.question_type in (
                QuestionType.MULTIPLE_CHOICE.value,
                QuestionType.TRUE_FALSE.value,
            ):
                is_correct, score, feedback_text = grade_objective_answer(
                    question_type=q.question_type,
                    student_answer=student_ans_str,
                    correct_answer=q.correct_answer,
                )
                evaluator = EvaluatorType.DETERMINISTIC.value
            else:
                evaluator = EvaluatorType.AI_ASSISTED.value
                score, is_correct, feedback_text = subjective_results[q.id]

            total_score += score
            graded_items.append(
                (q.id, student_ans_str, is_correct, score, evaluator, feedback_text)
            )

        # Step 3: Transactional atomic commit under row lock
        attempt = self.db.scalar(
            select(ExamAttempt)
            .options(
                selectinload(ExamAttempt.reviewer).selectinload(ExamReviewer.questions),
                selectinload(ExamAttempt.answers),
            )
            .where(
                ExamAttempt.id == attempt_id,
                ExamAttempt.user_id == user_id,
            )
            .with_for_update()
        )
        if attempt is None:
            raise NotFoundException("Exam attempt not found", code="ATTEMPT_NOT_FOUND")
        if attempt.status == AttemptStatus.COMPLETED.value:
            raise ConflictException(
                "This exam attempt has already been submitted and graded.",
                code="ATTEMPT_ALREADY_COMPLETED",
            )

        result_items: list[StudentAnswerResultRead] = []
        questions_by_id = {q.id: q for q in reviewer.questions}

        for q_id, s_ans, is_corr, sc, eval_t, fb in graded_items:
            q = questions_by_id[q_id]
            saved_ans = self.repo.save_student_answer(
                attempt_id=attempt.id,
                question_id=q.id,
                student_answer=s_ans,
                is_correct=is_corr,
                score=sc,
                evaluator=eval_t,
                feedback=fb,
            )
            result_items.append(
                StudentAnswerResultRead(
                    id=saved_ans.id,
                    question_id=q.id,
                    question_index=q.question_index,
                    question_type=q.question_type,
                    prompt=q.prompt,
                    options=q.options,
                    student_answer=s_ans,
                    correct_answer=q.correct_answer,
                    is_correct=is_corr,
                    score=sc,
                    max_score=1.0,
                    evaluator=eval_t,
                    evaluator_type=eval_t,
                    explanation=q.explanation,
                    feedback=fb,
                    source_reference=q.source_reference,
                )
            )

        percentage = round((total_score / max(1.0, max_score)) * 100.0, 1)
        if percentage >= 80.0:
            praise = "Excellent mastery of the material!"
        else:
            praise = (
                "Good effort! Review the detailed explanations and source citations below "
                "to reinforce missed topics."
            )
        feedback_summary = (
            f"You scored {total_score:.1f} out of {max_score:.0f} ({percentage}%). {praise}"
        )

        self.repo.complete_attempt(
            attempt,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            feedback_summary=feedback_summary,
        )
        self.db.commit()

        result_items.sort(key=lambda item: item.question_index)
        return ExamResultRead(
            attempt_id=attempt.id,
            reviewer_id=reviewer.id,
            reviewer_title=reviewer.title,
            attempt_number=attempt.attempt_number,
            status=attempt.status,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            feedback_summary=feedback_summary,
            started_at=attempt.started_at,
            completed_at=attempt.completed_at,
            answers=result_items,
        )

    def get_attempt_results(
        self,
        user_id: uuid.UUID,
        attempt_id: uuid.UUID,
    ) -> ExamResultRead:
        attempt = self.repo.get_attempt_for_user(attempt_id, user_id)
        if attempt is None:
            raise NotFoundException("Exam attempt not found", code="ATTEMPT_NOT_FOUND")

        if attempt.status != AttemptStatus.COMPLETED.value:
            raise BadRequestException(
                "This exam attempt has not been submitted yet.",
                code="ATTEMPT_IN_PROGRESS",
            )

        reviewer = attempt.reviewer
        result_items: list[StudentAnswerResultRead] = []

        for ans in attempt.answers:
            q = ans.question
            result_items.append(
                StudentAnswerResultRead(
                    id=ans.id,
                    question_id=q.id,
                    question_index=q.question_index,
                    question_type=q.question_type,
                    prompt=q.prompt,
                    options=q.options,
                    student_answer=ans.student_answer,
                    correct_answer=q.correct_answer,
                    is_correct=ans.is_correct,
                    score=ans.score,
                    max_score=1.0,
                    evaluator=ans.evaluator,
                    evaluator_type=ans.evaluator,
                    explanation=q.explanation,
                    feedback=ans.feedback,
                    source_reference=q.source_reference,
                )
            )

        result_items.sort(key=lambda item: item.question_index)

        return ExamResultRead(
            attempt_id=attempt.id,
            reviewer_id=reviewer.id,
            reviewer_title=reviewer.title,
            attempt_number=attempt.attempt_number,
            status=attempt.status,
            total_score=attempt.total_score or 0.0,
            max_score=attempt.max_score or float(len(result_items)),
            percentage=attempt.percentage or 0.0,
            feedback_summary=attempt.feedback_summary,
            started_at=attempt.started_at,
            completed_at=attempt.completed_at,
            answers=result_items,
        )
