import uuid
from collections.abc import Sequence

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from tests.conftest import RecordingAnalysisQueue
from tests.integration.test_processing_and_retrieval import DeterministicKeywordEmbeddingProvider

from app.modules.analysis.exceptions import AnalysisTransientError
from app.modules.analysis.models import AnalysisStatus, DocumentAnalysis
from app.modules.analysis.schemas import (
    AnalysisConcept,
    AnalysisDefinition,
    AnalysisExample,
    AnalysisSourceReference,
    AnalysisSubtopic,
    AnalysisTopic,
    ConceptRelationship,
    StructuredAnalysisResult,
)
from app.modules.analysis.service import AnalysisExecutionService
from app.modules.documents.models import ProcessingStatus
from app.modules.documents.storage import LocalStorageProvider
from app.modules.processing.chunking import ChunkingService
from app.modules.processing.retrieval import RetrievalService, RetrievedChunk
from app.modules.processing.service import DocumentProcessingService


class DeterministicAnalysisGenerator:
    """Mock analysis generator for integration testing without external LLM API calls."""

    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.generated_calls: list[dict[str, object]] = []

    def generate_analysis(
        self,
        *,
        chunks: Sequence[RetrievedChunk],
        focus_topic: str | None = None,
        coverage_note: str | None = None,
    ) -> StructuredAnalysisResult:
        self.generated_calls.append(
            {
                "chunks_count": len(chunks),
                "focus_topic": focus_topic,
                "coverage_note": coverage_note,
            }
        )
        if self.should_fail:
            raise AnalysisTransientError(
                "Simulated Gemini rate limit during analysis",
                code="ANALYSIS_RATE_LIMITED",
                user_message="AI rate limit reached. Please tap Retry Analysis.",
            )

        source_refs: list[AnalysisSourceReference] = []
        for idx, chunk in enumerate(chunks, start=1):
            source_refs.append(
                AnalysisSourceReference(
                    source_key=f"S{idx}",
                    document_id=chunk.document_id,
                    original_filename=chunk.original_filename,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    excerpt=chunk.text[:80],
                )
            )

        title = f"Analysis of {focus_topic}" if focus_topic else "Comprehensive Study Guide"
        return StructuredAnalysisResult(
            title=title,
            overview=f"Comprehensive overview generated from {len(chunks)} chunks.",
            coverage_note=coverage_note,
            unestablished_notes=None,
            main_topics=[
                AnalysisTopic(
                    title="Core Principles",
                    summary="Summary of the main topic.",
                    subtopics=[
                        AnalysisSubtopic(
                            title="Detailed Breakdown",
                            explanation="Explanation grounded in S1.",
                            source_keys=["S1"] if chunks else [],
                        )
                    ],
                    source_keys=["S1"] if chunks else [],
                )
            ],
            important_concepts=[
                AnalysisConcept(
                    name="Key Concept",
                    detailed_explanation="Explanation of key concept.",
                    source_keys=["S1"] if chunks else [],
                )
            ],
            definitions=[
                AnalysisDefinition(
                    term="Key Concept",
                    definition="Definition of key concept.",
                    source_keys=["S1"] if chunks else [],
                )
            ],
            examples=[
                AnalysisExample(
                    title="Real-world application",
                    description="Practical example of concept.",
                    source_keys=["S1"] if chunks else [],
                )
            ],
            concept_relationships=[
                ConceptRelationship(
                    from_concept="Core Principles",
                    to_concept="Key Concept",
                    relationship="Foundational base",
                    source_keys=["S1"] if chunks else [],
                )
            ],
            key_takeaways=["Essential takeaway 1", "Essential takeaway 2"],
            source_references=source_refs,
        )


def _register_user(client: TestClient, email: str) -> dict[str, str]:
    res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "Password123"},
    )
    assert res.status_code == 201
    token = res.json()["data"]["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_ready_doc(
    client: TestClient,
    headers: dict[str, str],
    filename: str,
    content: str,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> uuid.UUID:
    res = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )
    assert res.status_code == 201
    doc_id = uuid.UUID(res.json()["data"]["id"])

    processor = DocumentProcessingService(
        db_session,
        storage=storage_provider,
        embedding_provider=DeterministicKeywordEmbeddingProvider(),
        chunker=ChunkingService(chunk_size=100, chunk_overlap=20),
    )
    processed = processor.process_document(doc_id)
    assert processed is not None
    assert processed.processing_status == ProcessingStatus.READY.value
    return doc_id


def test_full_analysis_lifecycle_single_and_multi_document(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
    analysis_queue: RecordingAnalysisQueue,
) -> None:
    alice_headers = _register_user(client, "alice.analysis@example.com")
    doc1_id = _create_ready_doc(
        client,
        alice_headers,
        "Physics_Thermodynamics.txt",
        "The laws of thermodynamics govern entropy and energy conservation.",
        db_session,
        storage_provider,
    )
    doc2_id = _create_ready_doc(
        client,
        alice_headers,
        "Biology_Respiration.txt",
        "Cellular respiration in mitochondria produces ATP via glycolysis.",
        db_session,
        storage_provider,
    )

    # 1. Create multi-document analysis
    create_res = client.post(
        "/api/v1/analyses",
        headers=alice_headers,
        json={
            "document_ids": [str(doc1_id), str(doc2_id)],
            "focus_topic": "Energy Transformations",
        },
    )
    assert create_res.status_code == 201
    create_data = create_res.json()["data"]
    analysis_id = uuid.UUID(create_data["id"])
    assert create_data["status"] == "queued"
    assert create_data["focus_topic"] == "Energy Transformations"
    assert len(create_data["source_documents"]) == 2
    assert analysis_id in analysis_queue.enqueued_analysis_ids

    # 2. Execute analysis background job
    mock_generator = DeterministicAnalysisGenerator()
    exec_service = AnalysisExecutionService(
        db_session,
        retrieval_service=RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        ),
        generator=mock_generator,
    )
    completed_analysis = exec_service.execute_analysis(analysis_id)
    assert completed_analysis is not None
    assert completed_analysis.status == AnalysisStatus.COMPLETED.value
    assert completed_analysis.completed_at is not None
    assert completed_analysis.result_json is not None
    assert len(mock_generator.generated_calls) == 1

    # 3. GET /api/v1/analyses/{analysis_id} returns structured analysis and source references
    get_res = client.get(f"/api/v1/analyses/{analysis_id}", headers=alice_headers)
    assert get_res.status_code == 200
    detail = get_res.json()["data"]
    assert detail["status"] == "completed"
    assert detail["result"] is not None
    assert detail["result"]["title"] == "Analysis of Energy Transformations"
    assert len(detail["result"]["source_references"]) >= 2
    assert len(detail["result"]["main_topics"]) == 1
    assert len(detail["result"]["important_concepts"]) == 1
    assert len(detail["result"]["key_takeaways"]) == 2

    # 4. List analyses returns summary
    list_res = client.get("/api/v1/analyses", headers=alice_headers)
    assert list_res.status_code == 200
    list_data = list_res.json()["data"]
    assert list_data["total"] == 1
    assert list_data["items"][0]["id"] == str(analysis_id)
    assert list_data["items"][0]["status"] == "completed"
    assert len(list_data["items"][0]["source_documents"]) == 2

    # 5. Delete analysis cascades associations
    del_res = client.delete(f"/api/v1/analyses/{analysis_id}", headers=alice_headers)
    assert del_res.status_code == 200
    assert del_res.json()["data"]["deleted"] is True

    # Confirm deleted
    assert client.get(f"/api/v1/analyses/{analysis_id}", headers=alice_headers).status_code == 404
    links_count = db_session.scalar(
        select(DocumentAnalysis).where(DocumentAnalysis.id == analysis_id)
    )
    assert links_count is None


def test_analysis_ownership_isolation_and_unready_document_rejection(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    alice_headers = _register_user(client, "alice.iso@example.com")
    bob_headers = _register_user(client, "bob.iso@example.com")

    alice_doc_id = _create_ready_doc(
        client,
        alice_headers,
        "Alice_Doc.txt",
        "Some Alice content.",
        db_session,
        storage_provider,
    )

    # 1. Bob cannot create analysis using Alice's document -> 404 DOCUMENT_NOT_FOUND
    bob_create_res = client.post(
        "/api/v1/analyses",
        headers=bob_headers,
        json={"document_ids": [str(alice_doc_id)]},
    )
    assert bob_create_res.status_code == 404
    assert bob_create_res.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    # 2. Cannot create analysis with unready document -> 400 DOCUMENT_NOT_READY
    unready_up = client.post(
        "/api/v1/documents",
        headers=alice_headers,
        files={"file": ("Unready.txt", b"Content not yet processed", "text/plain")},
    )
    unready_doc_id = uuid.UUID(unready_up.json()["data"]["id"])

    unready_create_res = client.post(
        "/api/v1/analyses",
        headers=alice_headers,
        json={"document_ids": [str(unready_doc_id)]},
    )
    assert unready_create_res.status_code == 400
    assert unready_create_res.json()["error"]["code"] == "DOCUMENT_NOT_READY"

    # 3. Create analysis as Alice
    alice_analysis = client.post(
        "/api/v1/analyses",
        headers=alice_headers,
        json={"document_ids": [str(alice_doc_id)]},
    ).json()["data"]
    analysis_id = alice_analysis["id"]

    # Bob cannot GET, retry, or DELETE Alice's analysis
    assert client.get(f"/api/v1/analyses/{analysis_id}", headers=bob_headers).status_code == 404
    assert (
        client.post(f"/api/v1/analyses/{analysis_id}/retry", headers=bob_headers).status_code == 404
    )
    assert client.delete(f"/api/v1/analyses/{analysis_id}", headers=bob_headers).status_code == 404


def test_analysis_failure_retry_and_duplicate_protection(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
    analysis_queue: RecordingAnalysisQueue,
) -> None:
    headers = _register_user(client, "retry.tester@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "Sample.txt",
        "Photosynthesis converts sunlight into chemical energy.",
        db_session,
        storage_provider,
    )

    create_res = client.post(
        "/api/v1/analyses",
        headers=headers,
        json={"document_ids": [str(doc_id)]},
    )
    analysis_id = uuid.UUID(create_res.json()["data"]["id"])

    # 1. Execution fails with failing generator
    failing_gen = DeterministicAnalysisGenerator(should_fail=True)
    exec_service = AnalysisExecutionService(
        db_session,
        retrieval_service=RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        ),
        generator=failing_gen,
    )
    failed_analysis = exec_service.execute_analysis(analysis_id)
    assert failed_analysis is not None
    assert failed_analysis.status == AnalysisStatus.FAILED.value
    assert failed_analysis.error_code == "ANALYSIS_RATE_LIMITED"
    assert "rate limit reached" in (failed_analysis.error_message or "").lower()

    # 2. Successful retry enqueues job and resets error state
    retry_res = client.post(
        f"/api/v1/analyses/{analysis_id}/retry",
        headers=headers,
    )
    assert retry_res.status_code == 200
    assert retry_res.json()["data"]["status"] == "queued"
    assert retry_res.json()["data"]["error_code"] is None
    assert analysis_queue.enqueued_analysis_ids[-1] == analysis_id

    # 3. Successful execution with working generator
    working_gen = DeterministicAnalysisGenerator(should_fail=False)
    exec_service.generator = working_gen
    completed = exec_service.execute_analysis(analysis_id)
    assert completed is not None
    assert completed.status == AnalysisStatus.COMPLETED.value

    # 4. Retrying an already completed analysis returns 409 ANALYSIS_ALREADY_COMPLETED
    conflict_res = client.post(
        f"/api/v1/analyses/{analysis_id}/retry",
        headers=headers,
    )
    assert conflict_res.status_code == 409
    assert conflict_res.json()["error"]["code"] == "ANALYSIS_ALREADY_COMPLETED"
