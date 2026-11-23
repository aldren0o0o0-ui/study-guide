import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import BadRequestException, NotFoundException
from app.modules.analysis.models import AnalysisDocument, AnalysisStatus, DocumentAnalysis
from app.modules.analysis.schemas import (
    AnalysisConcept,
    AnalysisDefinition,
    AnalysisExample,
    AnalysisSourceReference,
    StructuredAnalysisResult,
)
from app.modules.documents.models import Document, DocumentChunk, ProcessingStatus
from app.modules.exams.exceptions import ExamPermanentError
from app.modules.exams.generator import (
    GeminiExamGenerator,
    RawExamReviewerSchema,
    RawGeneratedQuestion,
    build_exam_prompt,
    resolve_grounded_questions,
)
from app.modules.exams.models import ExamReviewer, ExamReviewerDocument, ExamStatus, QuestionType
from app.modules.exams.schemas import ExamCreateRequest
from app.modules.exams.service import ExamExecutionService, ExamReviewerService
from app.modules.processing.retrieval import RetrievedChunk


def _make_sample_chunk(key_idx: int = 1) -> tuple[str, RetrievedChunk]:
    doc_id = uuid.uuid4()
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        original_filename="Photosynthesis.pdf",
        chunk_index=key_idx - 1,
        page_start=key_idx,
        page_end=key_idx,
        page_number=key_idx,
        section_title=f"Section {key_idx}",
        text=(
            f"Light reactions convert water and solar energy into oxygen and ATP (Key S{key_idx})."
        ),
        distance=0.05,
        similarity_score=0.95,
    )
    return f"S{key_idx}", chunk


def _make_structured_analysis_result() -> StructuredAnalysisResult:
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    return StructuredAnalysisResult(
        title="Plant Biology & Photosynthesis",
        overview="Comprehensive analysis of photochemical light reactions and Calvin cycle.",
        important_concepts=[
            AnalysisConcept(
                name="Thylakoid Light Reaction",
                detailed_explanation="Photons excite chlorophyll pigments in photosystem II.",
                mechanism_steps=[
                    "Photon absorption",
                    "Water photolysis releasing oxygen",
                    "ATP synthesis",
                ],
                why_it_matters="Produces all atmospheric oxygen and foundational chemical energy.",
                source_keys=["S1"],
            )
        ],
        definitions=[
            AnalysisDefinition(
                term="Photolysis",
                definition="The decomposition or separation of molecules by the action of light.",
                source_keys=["S1"],
            )
        ],
        examples=[
            AnalysisExample(
                title="Sunlight Photolysis in Spinach Leaves",
                description="Oxygen bubbles form on submerged leaves exposed to solar radiation.",
                example_type="source_grounded",
                source_keys=["S1"],
            ),
            AnalysisExample(
                title="Solar Panel Analogy",
                description=(
                    "Imagine the thylakoid membrane is like a solar panel charging batteries."
                ),
                example_type="illustrative",
                source_keys=["S1"],
            ),
        ],
        source_references=[
            AnalysisSourceReference(
                source_key="S1",
                document_id=doc_id,
                original_filename="Photosynthesis.pdf",
                chunk_id=chunk_id,
                chunk_index=0,
                page_start=1,
                page_end=1,
                page_number=1,
                section_title="Light Reactions",
                excerpt="Light reactions convert water and solar energy into oxygen and ATP.",
            )
        ],
    )


def test_build_exam_prompt_with_analysis_filters_illustrative_examples() -> None:
    _, chunk = _make_sample_chunk(1)
    chunks_by_key = [("S1", chunk)]
    analysis_res = _make_structured_analysis_result()

    prompt = build_exam_prompt(
        chunks_by_key=chunks_by_key,
        question_count=5,
        focus_topic=None,
        allowed_types=[QuestionType.MULTIPLE_CHOICE],
        analysis_result=analysis_res,
    )

    # Must contain validated analysis concepts and definitions
    assert "<validated_analysis_content>" in prompt
    assert "Thylakoid Light Reaction" in prompt
    assert "Photolysis" in prompt
    assert "Sunlight Photolysis in Spinach Leaves" in prompt

    # MUST NOT contain illustrative analogy
    assert "Solar Panel Analogy" not in prompt
    assert "solar panel charging batteries" not in prompt

    # Must contain strict grounding directives
    assert "Never turn illustrative analogies" in prompt
    assert "</validated_analysis_content>" in prompt


def test_resolve_grounded_questions_strict_grounding_rejects_hallucinations() -> None:
    _, chunk1 = _make_sample_chunk(1)
    key_map = {"S1": chunk1}

    raw = RawExamReviewerSchema(
        title="Sample Quiz",
        questions=[
            RawGeneratedQuestion(
                question_type="multiple_choice",
                prompt="What excites chlorophyll in photosystem II?",
                options=["A. Photons", "B. Electrons", "C. Neutrons", "D. Glucose"],
                correct_answer="A",
                explanation="Photons of light excite chlorophyll electrons.",
                source_key="S1",
            ),
            RawGeneratedQuestion(
                question_type="true_false",
                prompt="Mitochondria perform the light reactions.",
                options=["True", "False"],
                correct_answer="False",
                explanation="Chloroplasts perform light reactions, not mitochondria.",
                source_key="S99",  # Invalid / hallucinated key!
            ),
        ],
    )

    # In strict_grounding mode, invalid question with S99 is discarded
    res = resolve_grounded_questions(raw, key_map=key_map, strict_grounding=True)
    assert len(res.questions) == 1
    assert res.questions[0].source_reference is not None
    assert res.questions[0].source_reference["source_key"] == "S1"

    # In non-strict mode (standalone), both questions remain
    res_standalone = resolve_grounded_questions(raw, key_map=key_map, strict_grounding=False)
    assert len(res_standalone.questions) == 2
    assert res_standalone.questions[1].source_reference is None


def test_resolve_grounded_questions_rejects_illustrative_analogy_questions() -> None:
    _, chunk1 = _make_sample_chunk(1)
    key_map = {"S1": chunk1}

    raw = RawExamReviewerSchema(
        title="Sample Quiz",
        questions=[
            RawGeneratedQuestion(
                question_type="short_answer",
                prompt=(
                    "According to the AI tutor analogy, how does a thylakoid "
                    "act like a solar panel?"
                ),
                correct_answer="It captures energy and converts it to chemical charge.",
                explanation="This is an illustrative analogy provided for intuition.",
                source_key="S1",
            ),
        ],
    )

    # Should discard question that tests illustrative analogies as factual truth
    with pytest.raises(ExamPermanentError) as exc_info:
        resolve_grounded_questions(
            raw,
            key_map=key_map,
            strict_grounding=True,
            min_required_questions=1,
        )
    assert exc_info.value.code == "EXAM_GROUNDING_FAILED"


def test_gemini_generator_bounded_regeneration_on_grounding_failure() -> None:
    _, chunk1 = _make_sample_chunk(1)
    analysis_res = _make_structured_analysis_result()

    calls = 0

    def _mock_call(prompt: str, model: str | None = None) -> RawExamReviewerSchema:
        nonlocal calls
        calls += 1
        if calls == 1:
            # First attempt produces ungrounded key S99
            return RawExamReviewerSchema(
                title="Photosynthesis Quiz",
                questions=[
                    RawGeneratedQuestion(
                        question_type="multiple_choice",
                        prompt="Bad Question?",
                        options=["A. 1", "B. 2", "C. 3", "D. 4"],
                        correct_answer="A",
                        explanation="Explanation",
                        source_key="S99",
                    )
                ],
            )
        # Second attempt (after bounded regeneration) produces valid key S1
        return RawExamReviewerSchema(
            title="Photosynthesis Quiz",
            questions=[
                RawGeneratedQuestion(
                    question_type="multiple_choice",
                    prompt="Good Question?",
                    options=["A. Valid", "B. Invalid", "C. Wrong", "D. Other"],
                    correct_answer="A",
                    explanation="Grounded Explanation",
                    source_key="S1",
                )
            ],
        )

    generator = GeminiExamGenerator(api_key="fake-key")
    generator._call_provider = _mock_call  # type: ignore[assignment]

    result = generator.generate_exam(
        chunks=[chunk1],
        question_count=1,
        analysis_result=analysis_res,
    )

    assert calls == 2  # Proves bounded regeneration ran!
    assert len(result.questions) == 1
    assert result.questions[0].source_reference is not None
    assert result.questions[0].source_reference["source_key"] == "S1"


def test_exam_reviewer_service_create_from_analysis_success() -> None:
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    analysis_id = uuid.uuid4()

    mock_db = MagicMock()
    mock_queue = MagicMock()

    analysis = DocumentAnalysis(
        id=analysis_id,
        user_id=user_id,
        title="Physics Analysis",
        status=AnalysisStatus.COMPLETED.value,
        result_json={"overview": "Physics summary", "main_topics": []},
    )
    analysis.analysis_documents = [
        AnalysisDocument(analysis_id=analysis_id, document_id=doc_id, position=0)
    ]

    doc = Document(
        id=doc_id,
        user_id=user_id,
        original_filename="Mechanics.pdf",
        storage_key=f"raw/{user_id}/Mechanics.pdf",
        file_extension="pdf",
        file_size_bytes=2048,
        checksum_sha256="hash123",
        processing_status=ProcessingStatus.READY.value,
    )

    # Mock DB queries
    mock_db.scalar.return_value = analysis
    mock_db.scalars.return_value.all.return_value = [doc]

    service = ExamReviewerService(mock_db, exam_queue=mock_queue)

    created_reviewer = ExamReviewer(
        id=uuid.uuid4(),
        user_id=user_id,
        analysis_id=analysis_id,
        title="Quiz: Physics Analysis",
        status=ExamStatus.QUEUED.value,
        question_count=10,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    created_reviewer.reviewer_documents = [
        ExamReviewerDocument(reviewer_id=created_reviewer.id, document_id=doc_id)
    ]
    created_reviewer.reviewer_documents[0].document = doc
    created_reviewer.questions = []
    created_reviewer.attempts = []

    service.repo.create_reviewer = MagicMock(return_value=created_reviewer)
    service.repo.get_reviewer_for_user = MagicMock(return_value=created_reviewer)

    payload = ExamCreateRequest(analysis_id=analysis_id)
    detail = service.create_reviewer(user_id=user_id, payload=payload)

    assert detail.analysis_id == analysis_id
    assert detail.title == "Quiz: Physics Analysis"
    assert mock_queue.enqueue_exam.called


def test_exam_reviewer_service_analysis_ownership_and_inaccessible_rejected() -> None:
    user_alice = uuid.uuid4()
    analysis_id = uuid.uuid4()

    mock_db = MagicMock()
    # Analysis not found or inaccessible under Alice's user_id
    mock_db.scalar.return_value = None

    service = ExamReviewerService(mock_db, exam_queue=MagicMock())
    payload = ExamCreateRequest(analysis_id=analysis_id)

    with pytest.raises(NotFoundException) as exc_info:
        service.create_reviewer(user_id=user_alice, payload=payload)
    assert exc_info.value.code == "ANALYSIS_NOT_FOUND"


def test_exam_reviewer_service_unready_analysis_rejected() -> None:
    user_id = uuid.uuid4()
    analysis_id = uuid.uuid4()

    mock_db = MagicMock()
    # Analysis is still processing
    unready_analysis = DocumentAnalysis(
        id=analysis_id,
        user_id=user_id,
        title="Processing Analysis",
        status=AnalysisStatus.PROCESSING.value,
        result_json=None,
    )
    mock_db.scalar.return_value = unready_analysis

    service = ExamReviewerService(mock_db, exam_queue=MagicMock())
    payload = ExamCreateRequest(analysis_id=analysis_id)

    with pytest.raises(BadRequestException) as exc_info:
        service.create_reviewer(user_id=user_id, payload=payload)
    assert exc_info.value.code == "ANALYSIS_NOT_COMPLETED"


def test_exam_reviewer_service_document_mismatch_rejected() -> None:
    user_id = uuid.uuid4()
    analysis_id = uuid.uuid4()
    doc_a = uuid.uuid4()
    foreign_doc_b = uuid.uuid4()

    mock_db = MagicMock()
    analysis = DocumentAnalysis(
        id=analysis_id,
        user_id=user_id,
        title="Physics Analysis",
        status=AnalysisStatus.COMPLETED.value,
        result_json={"overview": "Physics summary"},
    )
    analysis.analysis_documents = [
        AnalysisDocument(analysis_id=analysis_id, document_id=doc_a, position=0)
    ]
    mock_db.scalar.return_value = analysis

    service = ExamReviewerService(mock_db, exam_queue=MagicMock())
    # User provides document_ids referencing foreign_doc_b which is NOT in analysis
    payload = ExamCreateRequest(
        analysis_id=analysis_id,
        document_ids=[foreign_doc_b],
    )

    with pytest.raises(BadRequestException) as exc_info:
        service.create_reviewer(user_id=user_id, payload=payload)
    assert exc_info.value.code == "DOCUMENT_ANALYSIS_MISMATCH"


def test_standalone_exam_creation_remains_functional() -> None:
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    mock_db = MagicMock()
    mock_queue = MagicMock()
    doc = Document(
        id=doc_id,
        user_id=user_id,
        original_filename="Biology.pdf",
        storage_key=f"raw/{user_id}/Biology.pdf",
        file_extension="pdf",
        file_size_bytes=2048,
        checksum_sha256="hash123",
        processing_status=ProcessingStatus.READY.value,
    )
    mock_db.scalars.return_value.all.return_value = [doc]

    service = ExamReviewerService(mock_db, exam_queue=mock_queue)

    created_reviewer = ExamReviewer(
        id=uuid.uuid4(),
        user_id=user_id,
        analysis_id=None,
        title="Exam on Biology.pdf",
        status=ExamStatus.QUEUED.value,
        question_count=10,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    created_reviewer.reviewer_documents = [
        ExamReviewerDocument(reviewer_id=created_reviewer.id, document_id=doc_id)
    ]
    created_reviewer.reviewer_documents[0].document = doc
    created_reviewer.questions = []
    created_reviewer.attempts = []

    service.repo.create_reviewer = MagicMock(return_value=created_reviewer)
    service.repo.get_reviewer_for_user = MagicMock(return_value=created_reviewer)

    # Standalone request (no analysis_id)
    payload = ExamCreateRequest(document_ids=[doc_id])
    detail = service.create_reviewer(user_id=user_id, payload=payload)

    assert detail.analysis_id is None
    assert detail.title == "Exam on Biology.pdf"
    assert mock_queue.enqueue_exam.called


def test_execution_service_runs_analysis_grounded_generation() -> None:
    user_id = uuid.uuid4()
    reviewer_id = uuid.uuid4()
    analysis_id = uuid.uuid4()
    mock_db = MagicMock()
    mock_generator = MagicMock()
    mock_generator.generate_exam.return_value = MagicMock(title="Quiz: Plant Biology", questions=[])
    analysis_res = _make_structured_analysis_result()
    doc_id = analysis_res.source_references[0].document_id
    chunk_id = analysis_res.source_references[0].chunk_id
    analysis = DocumentAnalysis(
        id=analysis_id,
        user_id=user_id,
        title="Plant Biology",
        status=AnalysisStatus.COMPLETED.value,
        result_json=analysis_res.model_dump(mode="json"),
    )

    doc = Document(
        id=doc_id,
        user_id=user_id,
        original_filename="Photosynthesis.pdf",
        storage_key=f"raw/{user_id}/Photosynthesis.pdf",
        file_extension="pdf",
        file_size_bytes=2048,
        checksum_sha256="hash123",
        processing_status=ProcessingStatus.READY.value,
    )

    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        user_id=user_id,
        chunk_index=0,
        page_start=1,
        page_end=1,
        section_title="Light Reactions",
        content="Light reactions convert water and solar energy into oxygen and ATP.",
    )

    reviewer = ExamReviewer(
        id=reviewer_id,
        user_id=user_id,
        analysis_id=analysis_id,
        title="Quiz: Plant Biology",
        status=ExamStatus.QUEUED.value,
        question_count=5,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    link = ExamReviewerDocument(reviewer_id=reviewer_id, document_id=doc_id)
    link.document = doc
    reviewer.reviewer_documents = [link]

    # Return reviewer on first scalar query, analysis on second
    mock_db.scalar.side_effect = [reviewer, analysis, reviewer]
    mock_db.scalars.return_value.all.return_value = [chunk]

    service = ExamExecutionService(
        mock_db,
        generator=mock_generator,
        retrieval_service=MagicMock(),
    )

    service.execute_reviewer_generation(reviewer_id)

    # Verify generator was called with analysis_result
    assert mock_generator.generate_exam.called
    kwargs = mock_generator.generate_exam.call_args[1]
    assert kwargs["analysis_result"] is not None
    assert kwargs["analysis_result"].title == "Plant Biology & Photosynthesis"
