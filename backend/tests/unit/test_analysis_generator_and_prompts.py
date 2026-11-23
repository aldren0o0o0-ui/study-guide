import json
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors

from app.core.config import Settings
from app.modules.analysis.exceptions import (
    AnalysisGenerationError,
    AnalysisPermanentError,
    AnalysisTransientError,
)
from app.modules.analysis.generator import (
    GeminiAnalysisGenerator,
    assign_source_keys,
    build_analysis_prompt,
    resolve_and_ground_analysis_result,
)
from app.modules.analysis.queue import RQAnalysisQueue
from app.modules.analysis.schemas import (
    AnalysisConcept,
    AnalysisDefinition,
    AnalysisSubtopic,
    AnalysisTopic,
    LLMGeneratedAnalysisSchema,
)
from app.modules.analysis.service import select_chunks_within_budget
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
    doc_pdf_id = uuid.uuid4()
    doc_docx_id = uuid.uuid4()
    return [
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=doc_pdf_id,
            original_filename="Lecture_1_Thermodynamics.pdf",
            chunk_index=0,
            text=(
                "The First Law of Thermodynamics states that energy cannot be created or destroyed."
            ),
            distance=0.05,
            similarity_score=0.95,
            page_number=4,
            page_start=4,
            page_end=5,
            section_title=None,
        ),
        RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=doc_docx_id,
            original_filename="Cellular_Respiration_Guide.docx",
            chunk_index=1,
            text=(
                "Mitochondria synthesize ATP via oxidative phosphorylation across "
                "the inner membrane."
            ),
            distance=0.09,
            similarity_score=0.91,
            page_number=None,
            page_start=None,
            page_end=None,
            section_title="Chapter 2: ATP Synthesis",
        ),
    ]


def test_assign_source_keys_prompt_builder_and_server_side_grounding() -> None:
    chunks = _make_sample_chunks()
    ordered_chunks, key_map = assign_source_keys(chunks)
    assert list(key_map.keys()) == ["S1", "S2"]

    prompt = build_analysis_prompt(
        chunks_by_key=ordered_chunks,
        focus_topic="Energy conversion in physics and biology",
    )
    assert "Requested Focus Topic: Energy conversion in physics and biology" in prompt
    assert (
        '<source_chunk key="S1" filename="Lecture_1_Thermodynamics.pdf" '
        'chunk_index="0" pages="4-5">' in prompt
    )
    assert (
        '<source_chunk key="S2" filename="Cellular_Respiration_Guide.docx" chunk_index="1" '
        'section="Chapter 2: ATP Synthesis">' in prompt
    )

    # Simulate LLM output with valid keys ("S1", "s2") and a hallucinated key ("S99")
    raw_llm = LLMGeneratedAnalysisSchema(
        title="Energy Conservation and Cellular ATP Synthesis",
        overview=(
            "An integrated explanation of thermodynamic conservation and biological ATP production."
        ),
        unestablished_notes=None,
        main_topics=[
            AnalysisTopic(
                title="Thermodynamic Conservation",
                summary="Energy is conserved in isolated systems.",
                subtopics=[
                    AnalysisSubtopic(
                        title="First Law",
                        explanation="Total energy remains constant during transformations.",
                        source_keys=["S1", "S99"],
                    )
                ],
                source_keys=["S1", "INVALID_KEY"],
            )
        ],
        important_concepts=[
            AnalysisConcept(
                name="Oxidative Phosphorylation",
                detailed_explanation="ATP synthesis driven by proton motive force in mitochondria.",
                source_keys=["s2", "S2"],
            )
        ],
        definitions=[
            AnalysisDefinition(
                term="First Law of Thermodynamics",
                definition="Energy cannot be created or destroyed.",
                source_keys=["S1"],
            )
        ],
        examples=[],
        concept_relationships=[],
        key_takeaways=[
            "Energy conservation governs all physical and biological transformations.",
            "Mitochondria couple electron transport to ATP synthesis.",
        ],
    )

    grounded = resolve_and_ground_analysis_result(
        raw_llm,
        ordered_chunks=ordered_chunks,
        key_map=key_map,
        coverage_note="Synthesized from 2 chunks.",
    )

    assert grounded.title == "Energy Conservation and Cellular ATP Synthesis"
    assert grounded.coverage_note == "Synthesized from 2 chunks."
    # Hallucinated keys S99 and INVALID_KEY are stripped; lowercase "s2" is normalized to "S2"
    assert grounded.main_topics[0].source_keys == ["S1"]
    assert grounded.main_topics[0].subtopics[0].source_keys == ["S1"]
    assert grounded.important_concepts[0].source_keys == ["S2"]
    assert grounded.examples == []
    assert grounded.concept_relationships == []

    # Authoritative source_references come strictly from RetrievedChunk metadata
    assert len(grounded.source_references) == 2
    ref_s1 = grounded.source_references[0]
    assert ref_s1.source_key == "S1"
    assert ref_s1.document_id == chunks[0].document_id
    assert ref_s1.original_filename == "Lecture_1_Thermodynamics.pdf"
    assert ref_s1.page_number == 4
    assert ref_s1.page_start == 4
    assert ref_s1.page_end == 5
    assert ref_s1.section_title is None

    ref_s2 = grounded.source_references[1]
    assert ref_s2.source_key == "S2"
    assert ref_s2.document_id == chunks[1].document_id
    assert ref_s2.original_filename == "Cellular_Respiration_Guide.docx"
    assert ref_s2.page_number is None
    assert ref_s2.page_start is None
    assert ref_s2.page_end is None
    assert ref_s2.section_title == "Chapter 2: ATP Synthesis"


def test_gemini_analysis_generator_retries_rate_limits_and_validates_schema() -> None:
    chunks = _make_sample_chunks()
    sleep_delays: list[float] = []
    attempts = 0

    valid_payload = {
        "title": "Thermodynamics and ATP",
        "overview": "Grounded overview from S1 and S2.",
        "unestablished_notes": None,
        "main_topics": [
            {
                "title": "Conservation of Energy",
                "summary": "Energy cannot be created or destroyed.",
                "subtopics": [],
                "source_keys": ["S1"],
            }
        ],
        "important_concepts": [],
        "definitions": [],
        "examples": [],
        "concept_relationships": [],
        "key_takeaways": ["Energy is conserved."],
    }

    def _flaky_handler(*, model: str, contents: str, config: Any) -> _FakeGenerateResponse:
        _ = (model, contents, config)
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise genai_errors.APIError(429, {"error": {"message": "Quota exceeded"}})
        return _FakeGenerateResponse(parsed=None, text=json.dumps(valid_payload))

    generator = GeminiAnalysisGenerator(
        api_key="test-key-never-logged",
        model_name="gemini-2.5-flash",
        max_retries=3,
        initial_backoff_seconds=0.25,
        sleep_fn=sleep_delays.append,
        genai_client=_FakeGenAIClient(_flaky_handler),
    )

    result = generator.generate_analysis(chunks=chunks, focus_topic="Conservation")
    assert result.title == "Thermodynamics and ATP"
    assert len(result.source_references) == 1
    assert result.source_references[0].source_key == "S1"
    assert attempts == 3
    assert sleep_delays == [0.25, 0.5]

    # Exhausting 429 retries raises AnalysisTransientError(ANALYSIS_RATE_LIMITED)
    always_429 = GeminiAnalysisGenerator(
        api_key="test-key",
        max_retries=2,
        sleep_fn=lambda _: None,
        genai_client=_FakeGenAIClient(
            lambda **_: (_ for _ in ()).throw(
                genai_errors.APIError(429, {"error": {"message": "Rate limit"}})
            )
        ),
    )
    with pytest.raises(AnalysisTransientError) as exc_info:
        always_429.generate_analysis(chunks=chunks)
    assert exc_info.value.code == "ANALYSIS_RATE_LIMITED"
    assert exc_info.value.retryable is True

    # Non-retryable 400 error raises AnalysisPermanentError immediately without retry
    perm_calls = 0

    def _perm_handler(**_: Any) -> Any:
        nonlocal perm_calls
        perm_calls += 1
        raise genai_errors.APIError(400, {"error": {"message": "Bad request"}})

    perm_gen = GeminiAnalysisGenerator(
        api_key="test-key",
        max_retries=3,
        sleep_fn=lambda _: None,
        genai_client=_FakeGenAIClient(_perm_handler),
    )
    with pytest.raises(AnalysisPermanentError) as perm_exc:
        perm_gen.generate_analysis(chunks=chunks)
    assert perm_exc.value.code == "ANALYSIS_PROVIDER_REJECTED"
    assert perm_calls == 1

    # Invalid JSON / schema mismatch raises INVALID_ANALYSIS_OUTPUT
    invalid_json_gen = GeminiAnalysisGenerator(
        api_key="test-key",
        genai_client=_FakeGenAIClient(
            lambda **_: _FakeGenerateResponse(parsed=None, text="{not valid json")
        ),
    )
    with pytest.raises(AnalysisPermanentError) as bad_json_exc:
        invalid_json_gen.generate_analysis(chunks=chunks)
    assert bad_json_exc.value.code == "INVALID_ANALYSIS_OUTPUT"

    # Missing API key raises ANALYSIS_API_KEY_MISSING
    no_key_gen = GeminiAnalysisGenerator(api_key=None)
    with pytest.raises(AnalysisPermanentError) as no_key_exc:
        no_key_gen.generate_analysis(chunks=chunks)
    assert no_key_exc.value.code == "ANALYSIS_API_KEY_MISSING"


def test_select_chunks_within_budget_balances_multi_document_coverage() -> None:
    doc1 = uuid.uuid4()
    doc2 = uuid.uuid4()

    def _chunk(doc_id: uuid.UUID, idx: int, length: int) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            original_filename=f"doc-{doc_id}.pdf",
            chunk_index=idx,
            text="A" * length,
            distance=0.1 + (idx * 0.05),
            similarity_score=0.9 - (idx * 0.05),
            page_number=idx + 1,
            page_start=idx + 1,
            page_end=idx + 1,
            section_title=None,
        )

    doc1_chunks = [_chunk(doc1, 0, 100), _chunk(doc1, 1, 100), _chunk(doc1, 2, 100)]
    doc2_chunks = [_chunk(doc2, 0, 100), _chunk(doc2, 1, 100)]

    # Budget of 350 chars allows 3 chunks total (round-robin across documents)
    selected, was_capped = select_chunks_within_budget(
        [(doc1, doc1_chunks), (doc2, doc2_chunks)],
        max_context_chars=350,
    )
    assert was_capped is True
    assert len(selected) == 3
    selected_doc_ids = [c.document_id for c in selected]
    assert doc1 in selected_doc_ids
    assert doc2 in selected_doc_ids
    # Chunks are never sliced mid-sentence
    assert all(len(c.text) == 100 for c in selected)


def test_rq_analysis_queue_enqueues_and_handles_redis_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_id = uuid.uuid4()
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

    monkeypatch.setattr("app.modules.analysis.queue.rq.Queue", _FakeQueue)

    queue_service = RQAnalysisQueue(
        settings=Settings(REDIS_URL="redis://localhost:6379/0", PROCESSING_EAGER_FALLBACK=False),
        redis_connection=fake_redis,
    )
    job_id = queue_service.enqueue_analysis(analysis_id)
    assert job_id == f"analysis-generate-{analysis_id}"
    assert len(enqueued_calls) == 1
    assert enqueued_calls[0]["args"] == (str(analysis_id),)

    class _FailingQueue:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("Redis connection refused")

    monkeypatch.setattr("app.modules.analysis.queue.rq.Queue", _FailingQueue)
    with pytest.raises(AnalysisGenerationError) as exc_info:
        queue_service.enqueue_analysis(analysis_id)
    assert exc_info.value.code == "QUEUE_UNAVAILABLE"
    assert exc_info.value.retryable is True
