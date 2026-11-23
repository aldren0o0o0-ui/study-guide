import uuid
from collections.abc import Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tests.conftest import RecordingExamQueue
from tests.integration.test_processing_and_retrieval import DeterministicKeywordEmbeddingProvider

from app.modules.documents.models import ProcessingStatus
from app.modules.documents.storage import LocalStorageProvider
from app.modules.exams.generator import (
    GeneratedExamReviewerResult,
    GroundedQuestionData,
)
from app.modules.exams.grader import SubjectiveGradeResult
from app.modules.exams.models import (
    AttemptStatus,
    ExamAttempt,
    ExamQuestion,
    ExamStatus,
)
from app.modules.exams.service import ExamExecutionService
from app.modules.processing.chunking import ChunkingService
from app.modules.processing.retrieval import RetrievalService, RetrievedChunk
from app.modules.processing.service import DocumentProcessingService


class DeterministicExamGenerator:
    """Mock exam generator for integration tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_exam(
        self,
        *,
        chunks: Sequence[RetrievedChunk],
        question_count: int = 10,
        focus_topic: str | None = None,
        allowed_types: Sequence[Any] | None = None,
    ) -> GeneratedExamReviewerResult:
        self.calls.append({"chunks": len(chunks), "count": question_count, "topic": focus_topic})

        q1 = GroundedQuestionData(
            question_index=0,
            question_type="multiple_choice",
            prompt="What is the First Law of Thermodynamics?",
            options=[
                "A. Energy cannot be created or destroyed",
                "B. Entropy decreases over time",
                "C. Absolute zero is reachable",
                "D. Heat only flows from cold to hot",
            ],
            correct_answer="A",
            explanation="The First Law is the law of energy conservation.",
            grading_rubric=None,
            source_reference={
                "source_key": "S1",
                "document_id": str(chunks[0].document_id),
                "original_filename": chunks[0].original_filename,
                "chunk_id": str(chunks[0].chunk_id),
                "chunk_index": 0,
                "page_number": 1,
                "page_start": 1,
                "page_end": 1,
                "section_title": None,
                "excerpt": chunks[0].text[:60],
            },
        )

        q2 = GroundedQuestionData(
            question_index=1,
            question_type="true_false",
            prompt="Energy can be created from nothing in isolated physical systems.",
            options=["True", "False"],
            correct_answer="False",
            explanation="Energy cannot be created from nothing.",
            grading_rubric=None,
            source_reference={
                "source_key": "S1",
                "document_id": str(chunks[0].document_id),
                "original_filename": chunks[0].original_filename,
                "chunk_id": str(chunks[0].chunk_id),
                "chunk_index": 0,
                "page_number": 1,
                "page_start": 1,
                "page_end": 1,
                "section_title": None,
                "excerpt": chunks[0].text[:60],
            },
        )

        q3 = GroundedQuestionData(
            question_index=2,
            question_type="short_answer",
            prompt="Explain the difference between energy conservation and entropy.",
            options=None,
            correct_answer=(
                "Conservation means total energy is constant; entropy measures disorder."
            ),
            explanation="Conservation states energy quantity remains constant.",
            grading_rubric="Must state total energy is constant and entropy measures disorder.",
            source_reference={
                "source_key": "S1",
                "document_id": str(chunks[0].document_id),
                "original_filename": chunks[0].original_filename,
                "chunk_id": str(chunks[0].chunk_id),
                "chunk_index": 0,
                "page_number": 1,
                "page_start": 1,
                "page_end": 1,
                "section_title": None,
                "excerpt": chunks[0].text[:60],
            },
        )

        return GeneratedExamReviewerResult(
            title="Thermodynamics Exam Reviewer",
            questions=[q1, q2, q3],
        )


class DeterministicSubjectiveGrader:
    """Mock subjective grader for open-ended answers."""

    def grade_short_answer(
        self,
        *,
        prompt: str,
        correct_answer: str,
        grading_rubric: str | None,
        source_excerpt: str | None,
        student_answer: str,
    ) -> SubjectiveGradeResult:
        if "constant" in student_answer.lower() and "entropy" in student_answer.lower():
            return SubjectiveGradeResult(
                score=1.0,
                is_correct=True,
                feedback="Excellent explanation of conservation and entropy.",
            )
        if "constant" in student_answer.lower() or "disorder" in student_answer.lower():
            return SubjectiveGradeResult(
                score=0.5,
                is_correct=True,
                feedback="Good start, but missing comparison of entropy.",
            )
        return SubjectiveGradeResult(
            score=0.0,
            is_correct=False,
            feedback="Incorrect or incomplete response.",
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


def test_full_exam_reviewer_lifecycle_anti_cheat_and_grading(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
    exam_queue: RecordingExamQueue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alice_headers = _register_user(client, "alice.exam@example.com")
    bob_headers = _register_user(client, "bob.exam@example.com")

    doc_id = _create_ready_doc(
        client,
        alice_headers,
        "Physics_Thermodynamics.txt",
        "The First Law of Thermodynamics states energy is conserved in isolated systems.",
        db_session,
        storage_provider,
    )

    # 1. Create exam reviewer
    create_res = client.post(
        "/api/v1/exams",
        headers=alice_headers,
        json={
            "document_ids": [str(doc_id)],
            "focus_topic": "Thermodynamics",
            "question_count": 10,
        },
    )
    assert create_res.status_code == 201
    data = create_res.json()["data"]
    reviewer_id = uuid.UUID(data["id"])
    assert data["status"] == "queued"
    assert reviewer_id in exam_queue.enqueued_exam_ids

    # 2. Run background worker execution
    exec_service = ExamExecutionService(
        db_session,
        retrieval_service=RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        ),
        generator=DeterministicExamGenerator(),
    )
    completed_reviewer = exec_service.execute_reviewer_generation(reviewer_id)
    assert completed_reviewer is not None
    assert completed_reviewer.status == ExamStatus.COMPLETED.value
    assert completed_reviewer.question_count == 3

    # 3. ANTI-CHEAT SECURITY CHECK: Start Attempt
    # Check that student question payload contains ZERO answer keys, explanations, or rubrics!
    start_res = client.post(
        f"/api/v1/exams/{reviewer_id}/attempts",
        headers=alice_headers,
    )
    assert start_res.status_code == 201
    attempt_data = start_res.json()["data"]
    attempt_id = uuid.UUID(attempt_data["attempt_id"])
    assert attempt_data["status"] == "in_progress"
    assert len(attempt_data["questions"]) == 3

    for q in attempt_data["questions"]:
        assert "correct_answer" not in q
        assert "explanation" not in q
        assert "grading_rubric" not in q
        assert "source_reference" not in q

    q_mcq_id = attempt_data["questions"][0]["id"]
    q_tf_id = attempt_data["questions"][1]["id"]
    q_short_id = attempt_data["questions"][2]["id"]

    # 4. Submit Answers for Grading
    # Mock subjective grader for short answer
    monkeypatch.setattr(
        "app.modules.exams.service.create_subjective_grader",
        lambda *args, **kwargs: DeterministicSubjectiveGrader(),
    )

    submit_res = client.post(
        f"/api/v1/exams/attempts/{attempt_id}/submit",
        headers=alice_headers,
        json={
            "answers": [
                {"question_id": q_mcq_id, "answer": "A. Energy cannot be created or destroyed"},
                {"question_id": q_tf_id, "answer": "False"},
                {
                    "question_id": q_short_id,
                    "answer": "Energy total is constant while entropy measures molecular disorder.",
                },
            ]
        },
    )
    assert submit_res.status_code == 200
    result_data = submit_res.json()["data"]
    assert result_data["status"] == "completed"
    assert result_data["total_score"] == 3.0
    assert result_data["max_score"] == 3.0
    assert result_data["percentage"] == 100.0

    ans1 = result_data["answers"][0]
    assert ans1["is_correct"] is True
    assert ans1["score"] == 1.0
    assert ans1["evaluator"] == "deterministic"
    assert "First Law is the law" in ans1["explanation"]

    ans3 = result_data["answers"][2]
    assert ans3["is_correct"] is True
    assert ans3["score"] == 1.0
    assert ans3["evaluator"] == "ai_assisted"
    assert "Excellent explanation" in ans3["feedback"]
    assert ans3["source_reference"] is not None

    # 5. Duplicate submission rejection (409 Conflict)
    dup_res = client.post(
        f"/api/v1/exams/attempts/{attempt_id}/submit",
        headers=alice_headers,
        json={"answers": [{"question_id": q_mcq_id, "answer": "A"}]},
    )
    assert dup_res.status_code == 409
    assert dup_res.json()["error"]["code"] == "ATTEMPT_ALREADY_COMPLETED"

    # 6. Retrieve graded results via GET
    get_res = client.get(
        f"/api/v1/exams/attempts/{attempt_id}/results",
        headers=alice_headers,
    )
    assert get_res.status_code == 200
    assert get_res.json()["data"]["percentage"] == 100.0

    # 7. Cross-User Isolation: Bob cannot access Alice's reviewer, start attempt, or read results
    assert client.get(f"/api/v1/exams/{reviewer_id}", headers=bob_headers).status_code == 404
    assert (
        client.post(f"/api/v1/exams/{reviewer_id}/attempts", headers=bob_headers).status_code == 404
    )
    assert (
        client.get(f"/api/v1/exams/attempts/{attempt_id}/results", headers=bob_headers).status_code
        == 404
    )

    # 8. Deletion cascades
    del_res = client.delete(f"/api/v1/exams/{reviewer_id}", headers=alice_headers)
    assert del_res.status_code == 200
    assert client.get(f"/api/v1/exams/{reviewer_id}", headers=alice_headers).status_code == 404

    # Verify database cleanup
    rem_q = db_session.scalar(select(ExamQuestion).where(ExamQuestion.reviewer_id == reviewer_id))
    assert rem_q is None
    rem_att = db_session.scalar(select(ExamAttempt).where(ExamAttempt.reviewer_id == reviewer_id))
    assert rem_att is None


def test_generation_retry_clears_existing_questions_atomically(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers = _register_user(client, "retry.tester@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "Bio101.txt",
        "Cell theory states that all living organisms are composed of cells.",
        db_session,
        storage_provider,
    )

    create_res = client.post(
        "/api/v1/exams",
        headers=headers,
        json={"document_ids": [str(doc_id)], "question_count": 5},
    )
    reviewer_id = uuid.UUID(create_res.json()["data"]["id"])

    exec_service = ExamExecutionService(
        db_session,
        retrieval_service=RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        ),
        generator=DeterministicExamGenerator(),
    )

    # First generation
    rev1 = exec_service.execute_reviewer_generation(reviewer_id)
    assert rev1 is not None
    q_count_1 = db_session.scalar(
        select(func.count(ExamQuestion.id)).where(ExamQuestion.reviewer_id == reviewer_id)
    )
    assert q_count_1 == 3

    # Retry generation on same reviewer with pre-existing questions
    rev2 = exec_service.execute_reviewer_generation(reviewer_id)
    assert rev2 is not None
    assert rev2.status == ExamStatus.COMPLETED.value
    q_count_2 = db_session.scalar(
        select(func.count(ExamQuestion.id)).where(ExamQuestion.reviewer_id == reviewer_id)
    )
    assert q_count_2 == 3


def test_in_progress_attempt_retrieval_numbering_and_route_aliases(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _register_user(client, "attempt.tester@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "Physics.txt",
        (
            "Newton's laws of motion describe the relationship between a body "
            "and forces acting upon it."
        ),
        db_session,
        storage_provider,
    )

    create_res = client.post(
        "/api/v1/exams",
        headers=headers,
        json={"document_ids": [str(doc_id)], "question_count": 5},
    )
    reviewer_id = uuid.UUID(create_res.json()["data"]["id"])

    exec_service = ExamExecutionService(
        db_session,
        retrieval_service=RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        ),
        generator=DeterministicExamGenerator(),
    )
    exec_service.execute_reviewer_generation(reviewer_id)

    # Start Attempt 1
    att1_res = client.post(f"/api/v1/exams/{reviewer_id}/attempts", headers=headers)
    assert att1_res.status_code == 201
    att1_data = att1_res.json()["data"]
    assert att1_data["attempt_number"] == 1
    att1_id = att1_data["attempt_id"]

    # Safe retrieval of in-progress attempt with masked questions
    get_att1 = client.get(f"/api/v1/exams/attempts/{att1_id}", headers=headers)
    assert get_att1.status_code == 200
    assert get_att1.json()["data"]["attempt_number"] == 1
    assert len(get_att1.json()["data"]["questions"]) == 3
    for q in get_att1.json()["data"]["questions"]:
        assert "correct_answer" not in q
        assert "explanation" not in q

    # Start Attempt 2
    att2_res = client.post(f"/api/v1/exams/{reviewer_id}/attempts", headers=headers)
    assert att2_res.status_code == 201
    assert att2_res.json()["data"]["attempt_number"] == 2
    att2_id = att2_res.json()["data"]["attempt_id"]

    # Submit Attempt 2 via nested route alias
    monkeypatch.setattr(
        "app.modules.exams.service.create_subjective_grader",
        lambda *args, **kwargs: DeterministicSubjectiveGrader(),
    )
    q_ids = [q["id"] for q in att2_res.json()["data"]["questions"]]
    submit_res = client.post(
        f"/api/v1/exams/{reviewer_id}/attempts/{att2_id}/submit",
        headers=headers,
        json={"answers": [{"question_id": q_id, "answer": "A"} for q_id in q_ids]},
    )
    assert submit_res.status_code == 200
    assert submit_res.json()["data"]["attempt_number"] == 2

    # Get results via nested route alias
    res_alias = client.get(
        f"/api/v1/exams/{reviewer_id}/attempts/{att2_id}/results",
        headers=headers,
    )
    assert res_alias.status_code == 200
    assert res_alias.json()["data"]["attempt_number"] == 2


def test_gemini_failure_during_submission_does_not_fabricate_score(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _register_user(client, "outage.tester@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "Bio.txt",
        "Cell membranes consist of a lipid bilayer with embedded transport proteins.",
        db_session,
        storage_provider,
    )

    create_res = client.post(
        "/api/v1/exams",
        headers=headers,
        json={"document_ids": [str(doc_id)], "question_count": 5},
    )
    reviewer_id = uuid.UUID(create_res.json()["data"]["id"])

    exec_service = ExamExecutionService(
        db_session,
        retrieval_service=RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        ),
        generator=DeterministicExamGenerator(),
    )
    exec_service.execute_reviewer_generation(reviewer_id)

    att_res = client.post(f"/api/v1/exams/{reviewer_id}/attempts", headers=headers)
    att_id = att_res.json()["data"]["attempt_id"]
    q_short = att_res.json()["data"]["questions"][2]

    # Force subjective grader to throw an outage exception
    class FailingSubjectiveGrader:
        def grade_short_answer(self, **kwargs: object) -> object:
            raise RuntimeError("Gemini 503 Service Unavailable Simulated")

    monkeypatch.setattr(
        "app.modules.exams.service.create_subjective_grader",
        lambda *args, **kwargs: FailingSubjectiveGrader(),
    )

    sub_res = client.post(
        f"/api/v1/exams/attempts/{att_id}/submit",
        headers=headers,
        json={
            "answers": [
                {"question_id": q_short["id"], "answer": "Some long response over 10 chars"}
            ]
        },
    )
    # Must fail with 503 without fabricating 0.5 points
    assert sub_res.status_code == 503
    assert sub_res.json()["error"]["code"] == "AI_EVALUATION_UNAVAILABLE"

    # Verify attempt is still in_progress and has NOT been marked completed with fake points
    rem_att = db_session.scalar(select(ExamAttempt).where(ExamAttempt.id == uuid.UUID(att_id)))
    assert rem_att is not None
    assert rem_att.status == AttemptStatus.IN_PROGRESS.value
    assert rem_att.total_score is None
