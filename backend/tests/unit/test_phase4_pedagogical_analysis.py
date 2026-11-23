import uuid

from app.modules.analysis.generator import (
    ANALYSIS_SYSTEM_INSTRUCTION,
    AvailableAssetMeta,
    build_analysis_prompt,
    resolve_and_ground_analysis_result,
)
from app.modules.analysis.schemas import (
    AnalysisConcept,
    AnalysisExample,
    LLMGeneratedAnalysisSchema,
    StructuredAnalysisResult,
)
from app.modules.processing.retrieval import RetrievedChunk


def _make_dummy_chunk(key: str = "S1") -> tuple[str, RetrievedChunk]:
    doc_id = uuid.uuid4()
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        original_filename="Physics_101.pdf",
        chunk_index=0,
        page_start=1,
        page_end=1,
        page_number=1,
        section_title="Introduction to Mechanics",
        text="Newton's second law states that Force equals mass times acceleration (F=ma).",
        distance=0.1,
        similarity_score=0.9,
    )
    return key, chunk


def test_enhanced_analysis_concept_schema_and_defaults() -> None:
    # Full concept with new pedagogical fields
    concept = AnalysisConcept(
        name="Newton's Second Law",
        detailed_explanation=(
            "The acceleration of an object is directly proportional "
            "to the net force acting upon it."
        ),
        intuition=(
            "Pushing something harder makes it accelerate faster, "
            "but heavier objects are harder to accelerate."
        ),
        mechanism_steps=[
            "Apply a net external force to an object of mass m",
            "The force induces an acceleration vector a in the direction of the net force",
            "Calculate resulting motion using F = ma",
        ],
        why_it_matters="Foundation for all classical engineering and vehicle safety systems.",
        related_asset_ids=["asset-123"],
        source_keys=["S1"],
    )
    assert concept.intuition is not None
    assert len(concept.mechanism_steps) == 3
    assert concept.why_it_matters is not None
    assert concept.related_asset_ids == ["asset-123"]


def test_legacy_analysis_result_backward_compatibility() -> None:
    # Legacy result_json format without intuition, mechanism_steps, or example_type
    legacy_json = {
        "title": "Classical Mechanics Summary",
        "overview": "Overview of Newton's laws.",
        "coverage_note": None,
        "unestablished_notes": None,
        "main_topics": [],
        "important_concepts": [
            {
                "name": "Inertia",
                "detailed_explanation": (
                    "Tendency of an object to resist changes in its state of motion."
                ),
                "source_keys": ["S1"],
            }
        ],
        "definitions": [],
        "examples": [
            {
                "title": "Sliding Hockey Puck",
                "description": "A puck on frictionless ice continues moving indefinitely.",
                "source_keys": ["S1"],
            }
        ],
        "concept_relationships": [],
        "key_takeaways": ["Force causes acceleration"],
        "source_references": [],
    }

    parsed = StructuredAnalysisResult.model_validate(legacy_json)
    assert parsed.important_concepts[0].name == "Inertia"
    # New fields should default safely
    assert parsed.important_concepts[0].intuition is None
    assert parsed.important_concepts[0].mechanism_steps == []
    assert parsed.important_concepts[0].why_it_matters is None
    assert parsed.important_concepts[0].related_asset_ids == []
    assert parsed.examples[0].example_type == "source_grounded"


def test_example_classification_and_citation_isolation() -> None:
    key, chunk = _make_dummy_chunk("S1")
    key_map = {"S1": chunk}

    raw = LLMGeneratedAnalysisSchema(
        title="Physics Analysis",
        overview="Overview text",
        main_topics=[],
        important_concepts=[],
        definitions=[],
        examples=[
            AnalysisExample(
                title="Document Example: Apple Falling",
                description="Gravity pulls an apple to Earth as documented on page 1.",
                example_type="source_grounded",
                source_keys=["S1"],
            ),
            AnalysisExample(
                title="Illustrative Analogy: Stretching a Rubber Band",
                description="Think of potential energy like pulling back a rubber band.",
                example_type="illustrative",
                source_keys=["S1"],  # Illustrative examples should have citations cleared
            ),
        ],
        concept_relationships=[],
        key_takeaways=[],
    )

    result = resolve_and_ground_analysis_result(
        raw,
        ordered_chunks=[(key, chunk)],
        key_map=key_map,
    )

    assert len(result.examples) == 2
    ex1 = result.examples[0]
    assert ex1.example_type == "source_grounded"
    assert ex1.source_keys == ["S1"]

    ex2 = result.examples[1]
    assert ex2.example_type == "illustrative"
    # Illustrative analogies must not have source_keys
    assert ex2.source_keys == []


def test_server_validation_of_related_asset_ids() -> None:
    key, chunk = _make_dummy_chunk("S1")
    key_map = {"S1": chunk}

    valid_asset_id = str(uuid.uuid4())
    fake_hallucinated_id = str(uuid.uuid4())

    raw = LLMGeneratedAnalysisSchema(
        title="Optics Analysis",
        overview="Overview text",
        main_topics=[],
        important_concepts=[
            AnalysisConcept(
                name="Refraction",
                detailed_explanation="Bending of light passing through different media.",
                related_asset_ids=[valid_asset_id, fake_hallucinated_id],
                source_keys=["S1"],
            )
        ],
        definitions=[],
        examples=[],
        concept_relationships=[],
        key_takeaways=[],
    )

    result = resolve_and_ground_analysis_result(
        raw,
        ordered_chunks=[(key, chunk)],
        key_map=key_map,
        valid_asset_ids={valid_asset_id},
    )

    concept = result.important_concepts[0]
    assert valid_asset_id in concept.related_asset_ids
    assert fake_hallucinated_id not in concept.related_asset_ids


def test_build_analysis_prompt_with_available_figures() -> None:
    key, chunk = _make_dummy_chunk("S1")
    asset_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    assets = [
        AvailableAssetMeta(
            asset_id=asset_id,
            document_id=doc_id,
            page_number=3,
            caption="Free Body Diagram",
            section_title="Forces",
        )
    ]

    prompt = build_analysis_prompt(
        chunks_by_key=[(key, chunk)],
        focus_topic="Forces",
        available_assets=assets,
    )

    assert "<available_figures>" in prompt
    assert f'id="{asset_id}"' in prompt
    assert 'page="3"' in prompt
    assert 'caption="Free Body Diagram"' in prompt
    assert "<source_chunks>" in prompt


def test_prompt_system_instruction_contains_pedagogical_guidance() -> None:
    assert "PEDAGOGICAL STRUCTURE & INTUITION" in ANALYSIS_SYSTEM_INSTRUCTION
    assert "intuition" in ANALYSIS_SYSTEM_INSTRUCTION
    assert "mechanism_steps" in ANALYSIS_SYSTEM_INSTRUCTION
    assert "why_it_matters" in ANALYSIS_SYSTEM_INSTRUCTION
    assert "source_grounded" in ANALYSIS_SYSTEM_INSTRUCTION
    assert "illustrative" in ANALYSIS_SYSTEM_INSTRUCTION
