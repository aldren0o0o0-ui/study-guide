import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalysisCreateRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=10)
    focus_topic: str | None = Field(default=None, max_length=300)

    @field_validator("document_ids")
    @classmethod
    def deduplicate_document_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        unique_ids = list(dict.fromkeys(value))
        if not unique_ids:
            raise ValueError("At least one document must be selected")
        return unique_ids

    @field_validator("focus_topic", mode="before")
    @classmethod
    def normalize_focus_topic(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return value


class AnalysisSourceDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    file_extension: str
    processing_status: str


class AnalysisSourceReference(BaseModel):
    source_key: str = Field(..., min_length=1, max_length=16)
    document_id: uuid.UUID
    original_filename: str
    chunk_id: uuid.UUID
    chunk_index: int
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    excerpt: str


class AnalysisSubtopic(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    explanation: str = Field(..., min_length=1)
    source_keys: list[str] = Field(default_factory=list)


class AnalysisTopic(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    summary: str = Field(..., min_length=1)
    subtopics: list[AnalysisSubtopic] = Field(default_factory=list)
    source_keys: list[str] = Field(default_factory=list)


class AnalysisConcept(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)
    detailed_explanation: str = Field(..., min_length=1)
    intuition: str | None = Field(
        default=None,
        description="Plain-English, accessible summary of the concept's core idea.",
    )
    mechanism_steps: list[str] = Field(
        default_factory=list,
        description="Optional ordered steps explaining how the process works or derivation steps.",
    )
    why_it_matters: str | None = Field(
        default=None,
        description="Practical application, academic significance, or real-world relevance.",
    )
    related_asset_ids: list[str] = Field(
        default_factory=list,
        description="IDs of extracted original figures or diagrams that illustrate this concept.",
    )
    source_keys: list[str] = Field(default_factory=list)


class AnalysisDefinition(BaseModel):
    term: str = Field(..., min_length=1, max_length=300)
    definition: str = Field(..., min_length=1)
    source_keys: list[str] = Field(default_factory=list)


class AnalysisExample(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field(..., min_length=1)
    example_type: str = Field(
        default="source_grounded",
        description="Either 'source_grounded' (found in text) or 'illustrative' (analogy).",
    )
    source_keys: list[str] = Field(default_factory=list)


class ConceptRelationship(BaseModel):
    from_concept: str = Field(..., min_length=1, max_length=300)
    to_concept: str = Field(..., min_length=1, max_length=300)
    relationship: str = Field(..., min_length=1)
    source_keys: list[str] = Field(default_factory=list)


class LLMGeneratedAnalysisSchema(BaseModel):
    """Schema requested from the Gemini structured output generator."""

    title: str = Field(..., min_length=1, max_length=255)
    overview: str = Field(..., min_length=1)
    unestablished_notes: str | None = None
    main_topics: list[AnalysisTopic] = Field(default_factory=list)
    important_concepts: list[AnalysisConcept] = Field(default_factory=list)
    definitions: list[AnalysisDefinition] = Field(default_factory=list)
    examples: list[AnalysisExample] = Field(default_factory=list)
    concept_relationships: list[ConceptRelationship] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)


class StructuredAnalysisResult(BaseModel):
    """Validated, source-grounded analysis result persisted in `DocumentAnalysis.result_json`."""

    title: str
    overview: str
    coverage_note: str | None = None
    unestablished_notes: str | None = None
    main_topics: list[AnalysisTopic] = Field(default_factory=list)
    important_concepts: list[AnalysisConcept] = Field(default_factory=list)
    definitions: list[AnalysisDefinition] = Field(default_factory=list)
    examples: list[AnalysisExample] = Field(default_factory=list)
    concept_relationships: list[ConceptRelationship] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    source_references: list[AnalysisSourceReference] = Field(default_factory=list)


class AnalysisSummaryRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    focus_topic: str | None
    status: str
    source_documents: list[AnalysisSourceDocumentRead]
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class AnalysisDetailRead(AnalysisSummaryRead):
    result: StructuredAnalysisResult | None = None


class AnalysisListResponse(BaseModel):
    items: list[AnalysisSummaryRead]
    total: int
    limit: int
    offset: int


class AnalysisDeleteResponse(BaseModel):
    deleted: bool
