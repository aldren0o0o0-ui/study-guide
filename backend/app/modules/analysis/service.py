import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
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
from app.modules.analysis.exceptions import (
    AnalysisGenerationError,
    AnalysisPermanentError,
)
from app.modules.analysis.generator import (
    AnalysisGenerator,
    AvailableAssetMeta,
    create_analysis_generator,
)
from app.modules.analysis.models import AnalysisDocument, AnalysisStatus, DocumentAnalysis
from app.modules.analysis.repository import AnalysisRepository
from app.modules.analysis.schemas import (
    AnalysisCreateRequest,
    AnalysisDeleteResponse,
    AnalysisDetailRead,
    AnalysisListResponse,
    AnalysisSourceDocumentRead,
    AnalysisSummaryRead,
    StructuredAnalysisResult,
)
from app.modules.documents.models import Document, DocumentAsset, ProcessingStatus
from app.modules.processing.exceptions import DocumentProcessingError
from app.modules.processing.retrieval import RetrievalService, RetrievedChunk

logger = logging.getLogger("app.modules.analysis.service")


class AnalysisQueueDispatcher(Protocol):
    def enqueue_analysis(self, analysis_id: uuid.UUID) -> str: ...


def _build_initial_title(
    documents: Sequence[Document],
    focus_topic: str | None,
) -> str:
    if focus_topic:
        return f"Analysis: {focus_topic}"[:255]
    if len(documents) == 1:
        return f"Analysis of {documents[0].original_filename}"[:255]
    return f"Analysis of {documents[0].original_filename} + {len(documents) - 1} more"[:255]


def _to_source_docs_read(analysis: DocumentAnalysis) -> list[AnalysisSourceDocumentRead]:
    items: list[AnalysisSourceDocumentRead] = []
    for link in analysis.analysis_documents:
        if link.document is not None:
            items.append(AnalysisSourceDocumentRead.model_validate(link.document))
    return items


def _to_summary_read(analysis: DocumentAnalysis) -> AnalysisSummaryRead:
    return AnalysisSummaryRead(
        id=analysis.id,
        user_id=analysis.user_id,
        title=analysis.title,
        focus_topic=analysis.focus_topic,
        status=analysis.status,
        source_documents=_to_source_docs_read(analysis),
        error_code=analysis.error_code,
        error_message=analysis.error_message,
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        completed_at=analysis.completed_at,
    )


def _to_detail_read(analysis: DocumentAnalysis) -> AnalysisDetailRead:
    parsed_result: StructuredAnalysisResult | None = None
    if analysis.result_json is not None:
        parsed_result = StructuredAnalysisResult.model_validate(analysis.result_json)

    return AnalysisDetailRead(
        id=analysis.id,
        user_id=analysis.user_id,
        title=analysis.title,
        focus_topic=analysis.focus_topic,
        status=analysis.status,
        source_documents=_to_source_docs_read(analysis),
        result=parsed_result,
        error_code=analysis.error_code,
        error_message=analysis.error_message,
        created_at=analysis.created_at,
        updated_at=analysis.updated_at,
        completed_at=analysis.completed_at,
    )


def select_chunks_within_budget(
    chunks_by_doc: Sequence[tuple[uuid.UUID, Sequence[RetrievedChunk]]],
    *,
    max_context_chars: int,
) -> tuple[list[RetrievedChunk], bool]:
    """
    Selects retrieved chunks across one or more documents in round-robin order
    to ensure balanced multi-document coverage within `max_context_chars`.

    Never slices a chunk mid-sentence. Returns `(ordered_chunks, was_budget_capped)`.
    """
    doc_order = {doc_id: pos for pos, (doc_id, _) in enumerate(chunks_by_doc)}
    queues = [list(doc_chunks) for _, doc_chunks in chunks_by_doc if doc_chunks]
    if not queues:
        return [], False

    selected: list[RetrievedChunk] = []
    seen_chunk_ids: set[uuid.UUID] = set()
    total_chars = 0
    was_capped = False

    while any(queues):
        made_progress = False
        for q in queues:
            while q and q[0].chunk_id in seen_chunk_ids:
                q.pop(0)
            if not q:
                continue

            candidate = q[0]
            candidate_len = len(candidate.text)
            if selected and total_chars + candidate_len > max_context_chars:
                was_capped = True
                q.clear()
                continue

            q.pop(0)
            seen_chunk_ids.add(candidate.chunk_id)
            selected.append(candidate)
            total_chars += candidate_len
            made_progress = True

        if not made_progress:
            break

    # Sort selected chunks by document selection position and chunk_index for coherent reading order
    selected.sort(key=lambda c: (doc_order.get(c.document_id, 999), c.chunk_index))
    return selected, was_capped


class AnalysisExecutionService:
    """
    Executes the background RAG retrieval and Gemini structured analysis generation pipeline.
    """

    def __init__(
        self,
        db: Session,
        *,
        retrieval_service: RetrievalService | None = None,
        generator: AnalysisGenerator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.retrieval = retrieval_service or RetrievalService(db, settings=self.settings)
        self.generator = generator or create_analysis_generator(self.settings)

    def execute_analysis(self, analysis_id: uuid.UUID) -> DocumentAnalysis | None:
        stmt = (
            select(DocumentAnalysis)
            .options(
                selectinload(DocumentAnalysis.analysis_documents).selectinload(
                    AnalysisDocument.document
                )
            )
            .where(DocumentAnalysis.id == analysis_id)
            .with_for_update()
        )
        analysis = self.db.scalar(stmt)
        if analysis is None:
            self.db.rollback()
            logger.warning(
                "Analysis not found when starting execution job (may have been deleted)",
                extra={"extra_data": {"analysis_id": str(analysis_id)}},
            )
            return None

        if analysis.status == AnalysisStatus.PROCESSING.value:
            self.db.commit()
            logger.info(
                "Skipping duplicate analysis job execution; analysis is already processing",
                extra={"extra_data": {"analysis_id": str(analysis_id)}},
            )
            return analysis

        analysis.status = AnalysisStatus.PROCESSING.value
        analysis.error_code = None
        analysis.error_message = None
        self.db.commit()

        try:
            ready_docs: list[Document] = []
            for link in analysis.analysis_documents:
                doc = link.document
                if (
                    doc is not None
                    and doc.user_id == analysis.user_id
                    and doc.processing_status == ProcessingStatus.READY.value
                ):
                    ready_docs.append(doc)

            if not ready_docs:
                raise AnalysisPermanentError(
                    "No ready source documents remain associated with this analysis",
                    code="SOURCE_DOCUMENTS_UNAVAILABLE",
                    user_message=(
                        "The source documents for this analysis are no longer available or ready."
                    ),
                )

            selected_chunks, coverage_note = self._retrieve_and_budget_chunks(
                user_id=analysis.user_id,
                documents=ready_docs,
                focus_topic=analysis.focus_topic,
            )

            # Collect authorized visual assets for contextual figure correlation
            doc_ids = [doc.id for doc in ready_docs]
            db_assets = self.db.scalars(
                select(DocumentAsset)
                .where(
                    DocumentAsset.document_id.in_(doc_ids),
                    DocumentAsset.user_id == analysis.user_id,
                )
                .order_by(DocumentAsset.order_index.asc())
            ).all()

            available_assets = [
                AvailableAssetMeta(
                    asset_id=a.id,
                    document_id=a.document_id,
                    page_number=a.page_number,
                    caption=a.caption,
                    section_title=a.section_title,
                )
                for a in db_assets
            ]

            structured_result = self.generator.generate_analysis(
                chunks=selected_chunks,
                focus_topic=analysis.focus_topic,
                coverage_note=coverage_note,
                available_assets=available_assets,
            )

            return self._persist_completed_analysis(
                analysis_id=analysis_id,
                result=structured_result,
            )
        except AnalysisGenerationError as exc:
            return self._record_failure(
                analysis_id=analysis_id,
                error_code=exc.code,
                user_message=exc.user_message,
                exc=exc,
            )
        except DocumentProcessingError as exc:
            return self._record_failure(
                analysis_id=analysis_id,
                error_code=exc.code,
                user_message=exc.user_message,
                exc=exc,
            )
        except AppException as exc:
            return self._record_failure(
                analysis_id=analysis_id,
                error_code=exc.code,
                user_message=exc.message,
                exc=exc,
            )
        except Exception as exc:
            return self._record_failure(
                analysis_id=analysis_id,
                error_code="INTERNAL_ANALYSIS_ERROR",
                user_message=(
                    "An unexpected error occurred while generating this analysis. Please try again."
                ),
                exc=exc,
            )

    def _retrieve_and_budget_chunks(
        self,
        *,
        user_id: uuid.UUID,
        documents: Sequence[Document],
        focus_topic: str | None,
    ) -> tuple[list[RetrievedChunk], str | None]:
        doc_ids = [doc.id for doc in documents]
        top_k = max(1, min(50, self.settings.ANALYSIS_TOP_K_CHUNKS))
        per_doc_k = max(2, min(50, top_k // max(1, len(documents))))

        chunks_by_doc: list[tuple[uuid.UUID, Sequence[RetrievedChunk]]] = []

        if focus_topic:
            global_hits = self.retrieval.retrieve_relevant_chunks(
                user_id=user_id,
                query=focus_topic,
                document_ids=doc_ids,
                limit=top_k,
            )
            grouped: dict[uuid.UUID, list[RetrievedChunk]] = {doc.id: [] for doc in documents}
            for hit in global_hits:
                if hit.document_id in grouped:
                    grouped[hit.document_id].append(hit)

            # Ensure every selected document contributes at least its top chunks
            for doc in documents:
                if not grouped[doc.id]:
                    doc_hits = self.retrieval.retrieve_relevant_chunks(
                        user_id=user_id,
                        query=focus_topic,
                        document_ids=[doc.id],
                        limit=min(3, per_doc_k),
                    )
                    grouped[doc.id].extend(doc_hits)
                chunks_by_doc.append((doc.id, grouped[doc.id]))
        else:
            for doc in documents:
                aspect_queries = [
                    (
                        f"Core topics, foundational principles, key definitions, and terminology "
                        f"in {doc.original_filename}"
                    ),
                    (
                        "Key mechanisms, step-by-step processes, formulas, derivations, "
                        f"and methodology in {doc.original_filename}"
                    ),
                    (
                        "Practical applications, real-world examples, advanced implications, "
                        f"and key takeaways in {doc.original_filename}"
                    ),
                ]
                doc_hits = self.retrieval.retrieve_stratified_document_chunks(
                    user_id=user_id,
                    document_id=doc.id,
                    queries=aspect_queries,
                    total_limit=per_doc_k,
                    min_chunks_for_stratification=15,
                    num_strata=3,
                )
                chunks_by_doc.append((doc.id, doc_hits))

        selected_chunks, was_capped = select_chunks_within_budget(
            chunks_by_doc,
            max_context_chars=self.settings.ANALYSIS_MAX_CONTEXT_CHARS,
        )

        if not selected_chunks:
            raise AnalysisPermanentError(
                "No chunks could be retrieved from the selected documents",
                code="NO_SOURCE_CHUNKS_AVAILABLE",
                user_message="No readable content chunks were found in the selected documents.",
            )

        total_available_chunks = sum(max(0, doc.chunk_count) for doc in documents)
        coverage_note: str | None = None
        if was_capped or len(selected_chunks) < total_available_chunks:
            coverage_note = (
                f"Synthesized from the {len(selected_chunks)} most relevant source sections "
                f"(out of {total_available_chunks} total document chunks across "
                f"{len(documents)} document(s)) within the analysis context budget."
            )

        return selected_chunks, coverage_note

    def _persist_completed_analysis(
        self,
        *,
        analysis_id: uuid.UUID,
        result: StructuredAnalysisResult,
    ) -> DocumentAnalysis | None:
        stmt = (
            select(DocumentAnalysis)
            .options(
                selectinload(DocumentAnalysis.analysis_documents).selectinload(
                    AnalysisDocument.document
                )
            )
            .where(DocumentAnalysis.id == analysis_id)
            .with_for_update()
        )
        analysis = self.db.scalar(stmt)
        if analysis is None:
            self.db.rollback()
            logger.warning(
                "Analysis was deleted during generation; discarding result",
                extra={"extra_data": {"analysis_id": str(analysis_id)}},
            )
            return None

        analysis.title = result.title[:255]
        analysis.result_json = result.model_dump(mode="json")
        analysis.status = AnalysisStatus.COMPLETED.value
        analysis.completed_at = datetime.now(UTC)
        analysis.error_code = None
        analysis.error_message = None

        self.db.commit()
        self.db.refresh(analysis)

        logger.info(
            "Document analysis completed",
            extra={
                "extra_data": {
                    "analysis_id": str(analysis.id),
                    "user_id": str(analysis.user_id),
                    "source_references_count": len(result.source_references),
                }
            },
        )
        return analysis

    def _record_failure(
        self,
        *,
        analysis_id: uuid.UUID,
        error_code: str,
        user_message: str,
        exc: Exception,
    ) -> DocumentAnalysis | None:
        self.db.rollback()
        logger.warning(
            "Document analysis generation failed",
            extra={
                "extra_data": {
                    "analysis_id": str(analysis_id),
                    "error_code": error_code,
                    "exception_type": type(exc).__name__,
                }
            },
        )

        stmt = (
            select(DocumentAnalysis)
            .options(
                selectinload(DocumentAnalysis.analysis_documents).selectinload(
                    AnalysisDocument.document
                )
            )
            .where(DocumentAnalysis.id == analysis_id)
            .with_for_update()
        )
        analysis = self.db.scalar(stmt)
        if analysis is None:
            return None

        analysis.status = AnalysisStatus.FAILED.value
        analysis.error_code = error_code[:64]
        analysis.error_message = user_message[:500]
        analysis.result_json = None
        analysis.completed_at = None

        self.db.commit()
        self.db.refresh(analysis)
        return analysis


class AnalysisService:
    def __init__(
        self,
        db: Session,
        *,
        analysis_queue: AnalysisQueueDispatcher,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.repo = AnalysisRepository(db)
        self.analysis_queue = analysis_queue
        self.settings = settings or get_settings()

    def create_analysis(
        self,
        user_id: uuid.UUID,
        payload: AnalysisCreateRequest,
    ) -> AnalysisDetailRead:
        if len(payload.document_ids) > self.settings.ANALYSIS_MAX_DOCUMENTS:
            raise BadRequestException(
                f"Cannot select more than {self.settings.ANALYSIS_MAX_DOCUMENTS} documents",
                code="TOO_MANY_DOCUMENTS_SELECTED",
            )

        ordered_docs = self._validate_and_load_ready_documents(
            user_id=user_id,
            document_ids=payload.document_ids,
        )

        initial_title = _build_initial_title(ordered_docs, payload.focus_topic)
        analysis = self.repo.create_analysis(
            user_id=user_id,
            title=initial_title,
            focus_topic=payload.focus_topic,
            document_ids=[doc.id for doc in ordered_docs],
        )
        self.db.commit()

        refreshed = self.repo.get_analysis_for_user(analysis.id, user_id)
        assert refreshed is not None
        self._enqueue_on_create(refreshed)

        final_analysis = self.repo.get_analysis_for_user(analysis.id, user_id)
        assert final_analysis is not None
        return _to_detail_read(final_analysis)

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
                    f"Document '{doc.original_filename}' is not ready for analysis "
                    f"(current status: {doc.processing_status})",
                    code="DOCUMENT_NOT_READY",
                )
            ordered_docs.append(doc)

        return ordered_docs

    def _enqueue_on_create(self, analysis: DocumentAnalysis) -> None:
        try:
            self.analysis_queue.enqueue_analysis(analysis.id)
        except AnalysisGenerationError as exc:
            analysis.status = AnalysisStatus.FAILED.value
            analysis.error_code = exc.code[:64]
            analysis.error_message = exc.user_message[:500]
            self.db.commit()
        except Exception as exc:
            logger.warning(
                "Unexpected error enqueueing analysis job",
                extra={
                    "extra_data": {
                        "analysis_id": str(analysis.id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            analysis.status = AnalysisStatus.FAILED.value
            analysis.error_code = "QUEUE_UNAVAILABLE"
            analysis.error_message = (
                "Background analysis queue is temporarily unavailable. Please tap Retry Analysis."
            )
            self.db.commit()

    def list_analyses(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> AnalysisListResponse:
        items, total = self.repo.list_analyses_for_user(
            user_id,
            limit=limit,
            offset=offset,
        )
        return AnalysisListResponse(
            items=[_to_summary_read(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_analysis(
        self,
        user_id: uuid.UUID,
        analysis_id: uuid.UUID,
    ) -> AnalysisDetailRead:
        analysis = self.repo.get_analysis_for_user(analysis_id, user_id)
        if analysis is None:
            raise NotFoundException("Analysis not found", code="ANALYSIS_NOT_FOUND")
        return _to_detail_read(analysis)

    def retry_analysis(
        self,
        user_id: uuid.UUID,
        analysis_id: uuid.UUID,
    ) -> AnalysisDetailRead:
        analysis = self.repo.get_analysis_for_user(analysis_id, user_id)
        if analysis is None:
            raise NotFoundException("Analysis not found", code="ANALYSIS_NOT_FOUND")

        if analysis.status == AnalysisStatus.PROCESSING.value:
            raise ConflictException(
                "Analysis is currently being generated",
                code="ANALYSIS_ALREADY_PROCESSING",
            )
        if analysis.status == AnalysisStatus.COMPLETED.value:
            raise ConflictException(
                "Analysis has already completed",
                code="ANALYSIS_ALREADY_COMPLETED",
            )

        ready_docs = [
            link.document
            for link in analysis.analysis_documents
            if link.document is not None
            and link.document.user_id == user_id
            and link.document.processing_status == ProcessingStatus.READY.value
        ]
        if not ready_docs:
            raise BadRequestException(
                "Cannot retry analysis because its source documents are no longer available or "
                "ready",
                code="SOURCE_DOCUMENTS_UNAVAILABLE",
            )

        analysis.status = AnalysisStatus.QUEUED.value
        analysis.error_code = None
        analysis.error_message = None
        self.db.commit()

        try:
            self.analysis_queue.enqueue_analysis(analysis.id)
        except AnalysisGenerationError as exc:
            analysis.status = AnalysisStatus.FAILED.value
            analysis.error_code = exc.code[:64]
            analysis.error_message = exc.user_message[:500]
            self.db.commit()
            raise ServiceUnavailableException(
                exc.user_message,
                code=exc.code,
            ) from exc

        updated = self.repo.get_analysis_for_user(analysis.id, user_id)
        assert updated is not None
        return _to_detail_read(updated)

    def delete_analysis(
        self,
        user_id: uuid.UUID,
        analysis_id: uuid.UUID,
    ) -> AnalysisDeleteResponse:
        analysis = self.repo.get_analysis_for_user(analysis_id, user_id)
        if analysis is None:
            raise NotFoundException("Analysis not found", code="ANALYSIS_NOT_FOUND")

        self.repo.delete_analysis(analysis)
        self.db.commit()

        logger.info(
            "Document analysis deleted",
            extra={
                "extra_data": {
                    "user_id": str(user_id),
                    "analysis_id": str(analysis_id),
                }
            },
        )
        return AnalysisDeleteResponse(deleted=True)
