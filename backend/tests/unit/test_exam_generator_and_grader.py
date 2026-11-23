import json
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors

from app.core.config import Settings
from app.modules.exams.generator import (
    assign_source_keys,
    build_exam_prompt,
    resolve_grounded_questions,
)
from app.modules.exams.grader import (
    GeminiSubjectiveGrader,
    grade_objective_answer,
    normalize_objective_answer,
)
from app.modules.exams.models import QuestionType
from app.modules.exams.queue import RQExamQueue
from app.modules.processing.retrieval import RetrievedChunk


@dataclass
class _FakeGenerateResponse:
    parsed: Any = None
    text: str | None = None


class _FakeModelsClient:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._handler(model=model, contents=contents, config=config)


class _FakeGenAIClient:
    def __init__(self, handler: Any) -> None:
        self.models = _FakeModelsClient(handler)


def _make_sample_chunks() -> list[RetrievedChunk]:
    doc_id = uuid.uuid4()
    return [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            original_filename="Cellular_Respiration.pdf",
            chunk_index=0,
            text=(
                "Mitochondria generate ATP via oxidative phosphorylation across the inner membrane."
            ),
            distance=0.08,
            similarity_score=0.92,
            page_number=3,
            page_start=3,
            page_end=3,
            section_title=None,
        ),
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            original_filename="Cellular_Respiration.pdf",
            chunk_index=1,
            text="Glycolysis converts glucose into pyruvate in the cytoplasm, yielding 2 ATP net.",
            distance=0.12,
            similarity_score=0.88,
            page_number=5,
            page_start=5,
            page_end=6,
            section_title=None,
        ),
    ]


def test_assign_source_keys_prompt_builder_and_grounded_questions() -> None:
    chunks = _make_sample_chunks()
    ordered, key_map = assign_source_keys(chunks)
    assert list(key_map.keys()) == ["S1", "S2"]

    prompt = build_exam_prompt(
        chunks_by_key=ordered,
        question_count=5,
        focus_topic="Mitochondria",
        allowed_types=[QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE],
    )
    assert "Generate exactly 5 high-quality academic exam questions" in prompt
    assert "Target Focus Topic: Mitochondria" in prompt
    assert '<source_chunk key="S1" filename="Cellular_Respiration.pdf"' in prompt
    assert "Mitochondria generate ATP" in prompt

    raw_response = {
        "title": "Exam Reviewer: Cellular Respiration",
        "questions": [
            {
                "question_type": "multiple_choice",
                "prompt": "Where is ATP synthesized via oxidative phosphorylation?",
                "options": [
                    "A. Outer membrane",
                    "B. Inner mitochondrial membrane",
                    "C. Cytosol",
                    "D. Nucleus",
                ],
                "correct_answer": "B. Inner mitochondrial membrane",
                "explanation": "Oxidative phosphorylation occurs at the inner membrane.",
                "grading_rubric": None,
                "source_key": "S1",
            },
            {
                "question_type": "true_false",
                "prompt": "Glycolysis takes place in the cytoplasm.",
                "options": ["True", "False"],
                "correct_answer": "True",
                "explanation": "Glycolysis occurs in the cytoplasm, producing pyruvate.",
                "grading_rubric": None,
                "source_key": "S2",
            },
        ],
    }

    from app.modules.exams.generator import RawExamReviewerSchema

    validated_raw = RawExamReviewerSchema.model_validate(raw_response)
    result = resolve_grounded_questions(validated_raw, key_map=key_map)

    assert result.title == "Exam Reviewer: Cellular Respiration"
    assert len(result.questions) == 2

    q1 = result.questions[0]
    assert q1.question_type == "multiple_choice"
    assert q1.correct_answer == "B"  # Normalized to option letter
    assert q1.source_reference is not None
    assert q1.source_reference["source_key"] == "S1"
    assert q1.source_reference["page_number"] == 3

    q2 = result.questions[1]
    assert q2.question_type == "true_false"
    assert q2.correct_answer == "True"
    assert q2.source_reference is not None
    assert q2.source_reference["source_key"] == "S2"
    assert q2.source_reference["page_start"] == 5


def test_deterministic_objective_grading_normalization() -> None:
    # Multiple Choice
    assert normalize_objective_answer("B") == "B"
    assert normalize_objective_answer("b") == "B"
    assert normalize_objective_answer("B. Inner membrane") == "B"
    assert normalize_objective_answer(" (C) Outer membrane") == "C"

    # True / False
    assert normalize_objective_answer("true") == "True"
    assert normalize_objective_answer("T") == "True"
    assert normalize_objective_answer("FALSE") == "False"
    assert normalize_objective_answer("f") == "False"

    # Negative cases (prose responses must NOT false-positive match option letters)
    assert normalize_objective_answer("A single cell") == "A single cell"
    assert normalize_objective_answer("I don't know") == "I don't know"
    assert normalize_objective_answer("I think false") == "I think false"

    # Exact grading
    correct, score, _ = grade_objective_answer(
        question_type="multiple_choice",
        student_answer="B. Inner mitochondrial membrane",
        correct_answer="B",
    )
    assert correct is True
    assert score == 1.0

    incorrect, score_inc, msg = grade_objective_answer(
        question_type="multiple_choice",
        student_answer="A",
        correct_answer="B",
    )
    assert incorrect is False
    assert score_inc == 0.0
    assert "The correct answer is: B" in msg

    # Empty answer
    empty_c, empty_s, _ = grade_objective_answer(
        question_type="true_false",
        student_answer="  ",
        correct_answer="True",
    )
    assert empty_c is False
    assert empty_s == 0.0


def test_gemini_subjective_grader_handles_evaluation_and_retries() -> None:
    delays: list[float] = []
    attempts = 0

    def _flaky_handler(*, model: str, contents: str, config: Any) -> _FakeGenerateResponse:
        _ = (model, contents, config)
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise genai_errors.APIError(429, {"error": {"message": "Rate limit"}})
        payload = {
            "score": 0.85,
            "is_correct": True,
            "feedback": "Accurately described proton motive force and ATP synthase rotation.",
        }
        return _FakeGenerateResponse(parsed=None, text=json.dumps(payload))

    grader = GeminiSubjectiveGrader(
        api_key="test-key-never-logged",
        max_retries=3,
        initial_backoff_seconds=0.1,
        sleep_fn=delays.append,
        genai_client=_FakeGenAIClient(_flaky_handler),
    )

    result = grader.grade_short_answer(
        prompt="Explain ATP synthesis across inner membrane.",
        correct_answer="Protons drive ATP synthase rotation.",
        grading_rubric="Must mention proton gradient and ATP synthase.",
        source_excerpt="Mitochondria generate ATP via oxidative phosphorylation.",
        student_answer="Proton gradients flow across the membrane powering ATP synthase.",
    )

    assert result.score == 0.85
    assert result.is_correct is True
    assert "proton motive force" in result.feedback
    assert attempts == 2
    assert len(delays) == 1

    # Empty answer returns score 0 immediately without LLM call
    zero_res = grader.grade_short_answer(
        prompt="Explain ATP synthesis.",
        correct_answer="Protons drive ATP synthase.",
        grading_rubric=None,
        source_excerpt=None,
        student_answer="   ",
    )
    assert zero_res.score == 0.0
    assert zero_res.is_correct is False


def test_rq_exam_queue_enqueues_and_handles_redis_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer_id = uuid.uuid4()
    fake_redis = MagicMock()
    enqueued_calls: list[dict[str, Any]] = []

    class _FakeJob:
        def __init__(self, job_id: str) -> None:
            self.id = job_id

    class _FakeQueue:
        def __init__(self, name: str, connection: Any, default_timeout: int) -> None:
            self.name = name
            self.connection = connection
            self.default_timeout = default_timeout

        def enqueue_call(
            self,
            *,
            func: Any,
            args: tuple[Any, ...],
            job_id: str,
            timeout: int,
        ) -> _FakeJob:
            enqueued_calls.append(
                {"func": func, "args": args, "job_id": job_id, "timeout": timeout}
            )
            return _FakeJob(job_id)

    monkeypatch.setattr("app.modules.exams.queue.rq.Queue", _FakeQueue)

    queue = RQExamQueue(
        settings=Settings(REDIS_URL="redis://localhost:6379/0", PROCESSING_EAGER_FALLBACK=False),
        redis_connection=fake_redis,
    )
    job_id = queue.enqueue_exam(reviewer_id)
    assert job_id == f"exam-generate-{reviewer_id}"
    assert len(enqueued_calls) == 1
    assert enqueued_calls[0]["args"] == (str(reviewer_id),)
