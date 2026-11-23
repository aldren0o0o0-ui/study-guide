import uuid
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from tests.conftest import RecordingProcessingQueue
from tests.unit.test_extraction_and_chunking import (
    make_multipage_pdf_bytes,
    make_scanned_image_only_pdf_bytes,
    make_structured_docx_bytes,
)

from app.database.vector import DEFAULT_EMBEDDING_DIMENSION, cosine_distance_expression
from app.modules.documents.models import Document, DocumentChunk, ProcessingStatus
from app.modules.documents.storage import LocalStorageProvider
from app.modules.processing.chunking import ChunkingService
from app.modules.processing.exceptions import DocumentProcessingError, EmbeddingTransientError
from app.modules.processing.retrieval import RetrievalService
from app.modules.processing.service import DocumentProcessingService


class DeterministicKeywordEmbeddingProvider:
    """
    Deterministic test embedding provider that assigns orthogonal basis dimensions
    to domain keywords so PostgreSQL vector similarity rankings can be verified
    without paid external API calls.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [0.001] * self._dimension
        if "thermodynamics" in lowered or "entropy" in lowered:
            vec[0] = 1.0
        if "mitochondria" in lowered or "glycolysis" in lowered or "atp" in lowered:
            vec[1] = 1.0
        if "photosynthesis" in lowered or "chloroplast" in lowered:
            vec[2] = 1.0
        if "secret_alice_topic" in lowered:
            vec[3] = 1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)


class FailingEmbeddingProvider(DeterministicKeywordEmbeddingProvider):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        _ = texts
        raise EmbeddingTransientError(
            "Simulated Gemini rate limit during document embedding",
            code="EMBEDDING_RATE_LIMITED",
            user_message="AI rate limit reached while processing document. Please retry shortly.",
        )


def _register_user(client: TestClient, email: str) -> dict[str, str]:
    res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "Password123"},
    )
    assert res.status_code == 201
    token = res.json()["data"]["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_end_to_end_processing_retrieval_source_attribution_and_user_isolation(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
    processing_queue: RecordingProcessingQueue,
) -> None:
    alice_headers = _register_user(client, "alice.rag@example.com")
    bob_headers = _register_user(client, "bob.rag@example.com")

    embedding_provider = DeterministicKeywordEmbeddingProvider()
    processor = DocumentProcessingService(
        db_session,
        storage=storage_provider,
        embedding_provider=embedding_provider,
        chunker=ChunkingService(chunk_size=90, chunk_overlap=15),
    )
    retrieval = RetrievalService(
        db_session,
        embedding_provider=embedding_provider,
    )

    # 1. Alice uploads a multi-page PDF and a structured DOCX
    pdf_bytes = make_multipage_pdf_bytes(
        [
            "Lecture 1: Laws of Thermodynamics and energy conservation.",
            "Lecture 2: Entropy increases in isolated systems according to the Second Law.",
        ]
    )
    pdf_up = client.post(
        "/api/v1/documents",
        headers=alice_headers,
        files={"file": ("Physics_Thermodynamics.pdf", pdf_bytes, "application/pdf")},
    )
    assert pdf_up.status_code == 201
    pdf_doc_id = uuid.UUID(pdf_up.json()["data"]["id"])
    assert pdf_up.json()["data"]["processing_status"] == "queued"
    assert pdf_doc_id in processing_queue.enqueued_document_ids

    docx_bytes = make_structured_docx_bytes()
    docx_up = client.post(
        "/api/v1/documents",
        headers=alice_headers,
        files={
            "file": (
                "Biology_Respiration.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert docx_up.status_code == 201
    docx_doc_id = uuid.UUID(docx_up.json()["data"]["id"])

    # Before processing completes, unready documents are excluded from retrieval
    alice_user_id = uuid.UUID(pdf_up.json()["data"]["user_id"])
    assert (
        retrieval.retrieve_relevant_chunks(
            user_id=alice_user_id,
            query="thermodynamics and entropy",
        )
        == []
    )

    # 2. Run background processor for both documents
    processed_pdf = processor.process_document(pdf_doc_id)
    assert processed_pdf is not None
    assert processed_pdf.processing_status == ProcessingStatus.READY.value
    assert processed_pdf.chunk_count == 2
    assert processed_pdf.processed_at is not None
    assert processed_pdf.processing_error_code is None

    processed_docx = processor.process_document(docx_doc_id)
    assert processed_docx is not None
    assert processed_docx.processing_status == ProcessingStatus.READY.value
    assert processed_docx.chunk_count >= 2

    # Check status endpoint
    status_res = client.get(
        f"/api/v1/documents/{pdf_doc_id}/processing-status",
        headers=alice_headers,
    )
    assert status_res.status_code == 200
    status_data = status_res.json()["data"]
    assert status_data["processing_status"] == "ready"
    assert status_data["chunk_count"] == 2
    assert status_data["processed_at"] is not None

    # 3. Retrieve relevant chunks for thermodynamics query -> PDF chunks rank highest
    thermo_hits = retrieval.retrieve_relevant_chunks(
        user_id=alice_user_id,
        query="Explain entropy and thermodynamics",
        limit=3,
    )
    assert len(thermo_hits) >= 2
    top_hit = thermo_hits[0]
    assert top_hit.document_id == pdf_doc_id
    assert top_hit.original_filename == "Physics_Thermodynamics.pdf"
    assert top_hit.page_number in (1, 2)
    assert top_hit.page_start in (1, 2)
    assert top_hit.similarity_score > 0.9

    # 4. Retrieve relevant chunks for mitochondria/ATP query -> DOCX chunks rank highest
    bio_hits = retrieval.retrieve_relevant_chunks(
        user_id=alice_user_id,
        query="Where does glycolysis and mitochondria ATP synthesis happen?",
        limit=2,
    )
    assert len(bio_hits) == 2
    assert bio_hits[0].document_id == docx_doc_id
    assert bio_hits[0].original_filename == "Biology_Respiration.docx"
    assert bio_hits[0].page_number is None
    assert bio_hits[0].page_start is None
    assert bio_hits[0].page_end is None
    assert bio_hits[0].section_title == "Chapter 1: Cellular Respiration"

    # 5. Optional document_ids filter restricts search to specified documents
    restricted_hits = retrieval.retrieve_relevant_chunks(
        user_id=alice_user_id,
        query="Explain entropy and thermodynamics",
        document_ids=[docx_doc_id],
        limit=5,
    )
    assert all(hit.document_id == docx_doc_id for hit in restricted_hits)
    assert (
        retrieval.retrieve_relevant_chunks(
            user_id=alice_user_id,
            query="Explain entropy",
            document_ids=[],
        )
        == []
    )

    # 6. Strict cross-user isolation: Bob cannot retrieve Alice's chunks
    bob_me = client.get("/api/v1/auth/me", headers=bob_headers).json()["data"]
    bob_user_id = uuid.UUID(bob_me["id"])

    bob_attempt_all = retrieval.retrieve_relevant_chunks(
        user_id=bob_user_id,
        query="thermodynamics entropy mitochondria",
        limit=10,
    )
    assert bob_attempt_all == []

    bob_attempt_targeted = retrieval.retrieve_relevant_chunks(
        user_id=bob_user_id,
        query="thermodynamics entropy",
        document_ids=[pdf_doc_id, docx_doc_id],
        limit=10,
    )
    assert bob_attempt_targeted == []

    # Bob also cannot access Alice's processing-status or retry-processing endpoints
    assert (
        client.get(
            f"/api/v1/documents/{pdf_doc_id}/processing-status",
            headers=bob_headers,
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/v1/documents/{pdf_doc_id}/retry-processing",
            headers=bob_headers,
        ).status_code
        == 404
    )


def test_processing_failures_retries_duplicate_jobs_reprocessing_and_cascade_delete(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
    processing_queue: RecordingProcessingQueue,
) -> None:
    headers = _register_user(client, "lifecycle@example.com")
    embedding_provider = DeterministicKeywordEmbeddingProvider()

    # 1. Failed extraction (scanned image-only PDF without OCR text)
    scanned_pdf = make_scanned_image_only_pdf_bytes()
    scanned_up = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("Scanned_Handout.pdf", scanned_pdf, "application/pdf")},
    )
    assert scanned_up.status_code == 201
    scanned_id = uuid.UUID(scanned_up.json()["data"]["id"])

    processor = DocumentProcessingService(
        db_session,
        storage=storage_provider,
        embedding_provider=embedding_provider,
    )
    failed_doc = processor.process_document(scanned_id)
    assert failed_doc is not None
    assert failed_doc.processing_status == ProcessingStatus.FAILED.value
    assert failed_doc.processing_error_code == "OCR_NOT_SUPPORTED"
    assert failed_doc.processing_error_message is not None
    assert "OCR is not supported" in failed_doc.processing_error_message
    assert "Traceback" not in failed_doc.processing_error_message
    assert failed_doc.chunk_count == 0

    # 2. Failed embedding generation -> rolls back cleanly, marks failed, allows retry
    txt_up = client.post(
        "/api/v1/documents",
        headers=headers,
        files={
            "file": (
                "Photosynthesis_Notes.txt",
                b"Photosynthesis in chloroplasts converts light into glucose.",
                "text/plain",
            )
        },
    )
    assert txt_up.status_code == 201
    txt_id = uuid.UUID(txt_up.json()["data"]["id"])

    failing_processor = DocumentProcessingService(
        db_session,
        storage=storage_provider,
        embedding_provider=FailingEmbeddingProvider(),
    )
    embed_failed_doc = failing_processor.process_document(txt_id)
    assert embed_failed_doc is not None
    assert embed_failed_doc.processing_status == ProcessingStatus.FAILED.value
    assert embed_failed_doc.processing_error_code == "EMBEDDING_RATE_LIMITED"
    assert embed_failed_doc.chunk_count == 0

    chunk_rows_after_fail = db_session.scalar(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == txt_id)
    )
    assert chunk_rows_after_fail == 0

    # 3. Controlled retry via API transitions failed -> queued and enqueues job
    retry_res = client.post(
        f"/api/v1/documents/{txt_id}/retry-processing",
        headers=headers,
    )
    assert retry_res.status_code == 200
    retry_data = retry_res.json()["data"]
    assert retry_data["processing_status"] == "queued"
    assert retry_data["processing_error_code"] is None
    assert retry_data["processing_error_message"] is None
    assert processing_queue.enqueued_document_ids[-1] == txt_id

    # Now process with working provider -> transitions to ready
    ready_doc = processor.process_document(txt_id)
    assert ready_doc is not None
    assert ready_doc.processing_status == ProcessingStatus.READY.value
    assert ready_doc.chunk_count == 1

    # Retrying an already-ready document returns 409 DOCUMENT_ALREADY_READY
    conflict_ready = client.post(
        f"/api/v1/documents/{txt_id}/retry-processing",
        headers=headers,
    )
    assert conflict_ready.status_code == 409
    assert conflict_ready.json()["error"]["code"] == "DOCUMENT_ALREADY_READY"

    # 4. Reprocessing replaces stale chunks atomically without duplicate rows
    first_chunk_id = db_session.scalar(
        select(DocumentChunk.id).where(DocumentChunk.document_id == txt_id)
    )
    assert first_chunk_id is not None

    reprocessed_doc = processor.process_document(txt_id)
    assert reprocessed_doc is not None
    assert reprocessed_doc.processing_status == ProcessingStatus.READY.value
    chunks_after_reprocess = list(
        db_session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == txt_id)).all()
    )
    assert len(chunks_after_reprocess) == 1
    assert chunks_after_reprocess[0].id != first_chunk_id

    # 5. Duplicate job execution while document is already in 'processing' state is a safe no-op
    db_doc = db_session.scalar(select(Document).where(Document.id == txt_id))
    assert db_doc is not None
    db_doc.processing_status = ProcessingStatus.PROCESSING.value
    db_session.commit()

    dup_result = processor.process_document(txt_id)
    assert dup_result is not None
    assert dup_result.processing_status == ProcessingStatus.PROCESSING.value

    # Retrying while 'processing' returns 409 DOCUMENT_ALREADY_PROCESSING
    conflict_proc = client.post(
        f"/api/v1/documents/{txt_id}/retry-processing",
        headers=headers,
    )
    assert conflict_proc.status_code == 409
    assert conflict_proc.json()["error"]["code"] == "DOCUMENT_ALREADY_PROCESSING"

    # 6. Deleting document cascades and deletes all DocumentChunk rows
    del_res = client.delete(f"/api/v1/documents/{txt_id}", headers=headers)
    assert del_res.status_code == 200
    remaining_chunks = db_session.scalar(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == txt_id)
    )
    assert remaining_chunks == 0


def test_upload_records_failed_status_gracefully_when_queue_is_unavailable(
    client: TestClient,
    processing_queue: RecordingProcessingQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _register_user(client, "queue.down@example.com")

    def _raise_queue_down(_doc_id: uuid.UUID) -> str:
        raise DocumentProcessingError(
            "Redis connection refused",
            code="QUEUE_UNAVAILABLE",
            user_message="Background processing queue is temporarily unavailable.",
            retryable=True,
        )

    monkeypatch.setattr(processing_queue, "enqueue_document", _raise_queue_down)

    res = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("Offline_Queue.txt", b"Uploaded while Redis is down", "text/plain")},
    )
    assert res.status_code == 201
    data = res.json()["data"]
    assert data["processing_status"] == "failed"
    assert data["processing_error_code"] == "QUEUE_UNAVAILABLE"
    assert "temporarily unavailable" in data["processing_error_message"]


def test_pgvector_and_fallback_cosine_distance_sql_compilation() -> None:
    pg_expr = cosine_distance_expression(
        DocumentChunk.embedding,
        [1.0, 0.0, 0.0],
        use_pgvector=True,
    )
    pg_dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    pg_sql = str(
        pg_expr.compile(
            dialect=pg_dialect,
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "<=>" in pg_sql
    assert "[1.0,0.0,0.0]" in pg_sql

    fb_expr = cosine_distance_expression(
        DocumentChunk.embedding,
        [1.0, 0.0, 0.0],
        use_pgvector=False,
    )
    fb_sql = str(fb_expr.compile(dialect=pg_dialect))
    assert "sg_cosine_distance" in fb_sql
