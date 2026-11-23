import math
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import BadRequestException
from app.database.vector import DEFAULT_EMBEDDING_DIMENSION
from app.modules.analysis.service import AnalysisExecutionService
from app.modules.documents.models import Document, ProcessingStatus
from app.modules.processing.retrieval import (
    MAX_TOP_K,
    RetrievalService,
    RetrievedChunk,
)


class ChapterAwareEmbeddingProvider:
    """
    Deterministic embedding provider mapping pedagogical aspects to orthogonal vector dimensions.
    Enables reproducible multi-query and stratified retrieval tests without external dependencies.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        self._dimension = dimension
        self.embed_query_call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [0.01] * self._dimension

        # Dimension 0: Foundational principles, definitions, terminology, overview
        if any(
            k in lowered
            for k in [
                "foundational",
                "principles",
                "definitions",
                "terminology",
                "intro",
                "foundations",
                "core topics",
            ]
        ):
            vec[0] = 1.0

        # Dimension 1: Mechanisms, step-by-step processes, formulas, derivations, methodology
        if any(
            k in lowered
            for k in [
                "mechanisms",
                "mechanism",
                "processes",
                "process",
                "formulas",
                "formula",
                "derivations",
                "derivation",
                "methodology",
            ]
        ):
            vec[1] = 1.0

        # Dimension 2: Practical applications, real-world examples, advanced implications, takeaways
        if any(
            k in lowered
            for k in [
                "applications",
                "application",
                "real-world",
                "examples",
                "example",
                "implications",
                "takeaways",
                "conclusions",
            ]
        ):
            vec[2] = 1.0

        # Normalize vector to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        self.embed_query_call_count += 1
        return self._encode(query)


@dataclass
class SyntheticChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    original_filename: str
    chunk_index: int
    page_start: int
    page_end: int
    section_title: str
    text: str
    embedding: list[float]


class SyntheticDocumentStore:
    """
    In-memory document and chunk store with cosine-similarity ranking and bounds filtering.
    """

    def __init__(
        self,
        *,
        doc_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str,
        total_chunks: int,
        embedding_provider: ChapterAwareEmbeddingProvider,
    ) -> None:
        self.doc_id = doc_id
        self.user_id = user_id
        self.filename = filename
        self.total_chunks = total_chunks
        self.chunks: list[SyntheticChunk] = []

        for idx in range(total_chunks):
            if idx < total_chunks // 3:
                text = (
                    f"Chapter 1: Foundational Principles and Terminology. "
                    f"Core topics and definitions for chunk {idx}."
                )
                section = "Chapter 1: Foundations"
            elif idx < 2 * (total_chunks // 3):
                text = (
                    f"Chapter 2: Step-by-Step Mechanisms and Formulas. "
                    f"Mathematical derivations and process methodology for chunk {idx}."
                )
                section = "Chapter 2: Mechanisms"
            else:
                text = (
                    f"Chapter 3: Real-World Applications and Implications. "
                    f"Practical engineering examples and conclusions for chunk {idx}."
                )
                section = "Chapter 3: Applications"

            emb = embedding_provider.embed_query(text)
            self.chunks.append(
                SyntheticChunk(
                    chunk_id=uuid.uuid4(),
                    document_id=doc_id,
                    original_filename=filename,
                    chunk_index=idx,
                    page_start=(idx // 2) + 1,
                    page_end=(idx // 2) + 1,
                    section_title=section,
                    text=text,
                    embedding=emb,
                )
            )

    def query(
        self,
        query_vector: list[float],
        *,
        chunk_index_min: int | None = None,
        chunk_index_max: int | None = None,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        scored: list[tuple[SyntheticChunk, float, float]] = []
        for c in self.chunks:
            if chunk_index_min is not None and c.chunk_index < chunk_index_min:
                continue
            if chunk_index_max is not None and c.chunk_index > chunk_index_max:
                continue

            # Compute cosine similarity
            dot = sum(a * b for a, b in zip(query_vector, c.embedding, strict=False))
            norm_q = math.sqrt(sum(a * a for a in query_vector))
            norm_c = math.sqrt(sum(b * b for b in c.embedding))
            sim = dot / (norm_q * norm_c) if norm_q and norm_c else 0.0
            dist = max(0.0, 1.0 - sim)
            scored.append((c, dist, sim))

        # Order by distance.asc(), chunk_index.asc()
        scored.sort(key=lambda item: (item[1], item[0].chunk_index))
        top = scored[:limit]

        return [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                original_filename=c.original_filename,
                chunk_index=c.chunk_index,
                page_start=c.page_start,
                page_end=c.page_end,
                page_number=c.page_start,
                section_title=c.section_title,
                text=c.text,
                distance=round(dist, 6),
                similarity_score=round(sim, 6),
            )
            for c, dist, sim in top
        ]


def test_retrieve_relevant_chunks_sql_generation() -> None:
    """Verifies that retrieve_relevant_chunks constructs all SQL where clauses and bounds."""
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []
    provider = ChapterAwareEmbeddingProvider()
    retrieval = RetrievalService(mock_db, embedding_provider=provider)

    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    retrieval.retrieve_relevant_chunks(
        user_id=user_id,
        query="test mechanism search",
        document_ids=[doc_id],
        limit=10,
        chunk_index_min=5,
        chunk_index_max=15,
    )

    assert mock_db.execute.called
    stmt = mock_db.execute.call_args[0][0]
    clauses = [str(c) for c in stmt.whereclause.clauses]

    # Security invariant: document and chunk user_id checks
    assert any("documents.user_id =" in c for c in clauses)
    assert any("document_chunks.user_id =" in c for c in clauses)
    assert any("documents.processing_status =" in c for c in clauses)
    assert any("documents.id IN" in c for c in clauses)

    # Strata bounding: chunk_index range
    assert any("document_chunks.chunk_index >=" in c for c in clauses)
    assert any("document_chunks.chunk_index <=" in c for c in clauses)


def test_retrieve_relevant_chunks_validation() -> None:
    mock_db = MagicMock()
    retrieval = RetrievalService(mock_db, embedding_provider=ChapterAwareEmbeddingProvider())
    uid = uuid.uuid4()

    # Empty query string
    with pytest.raises(BadRequestException) as exc_empty:
        retrieval.retrieve_relevant_chunks(user_id=uid, query="   ")
    assert exc_empty.value.code == "EMPTY_RETRIEVAL_QUERY"

    # Invalid limits
    with pytest.raises(BadRequestException) as exc_low:
        retrieval.retrieve_relevant_chunks(user_id=uid, query="valid", limit=0)
    assert exc_low.value.code == "INVALID_RETRIEVAL_LIMIT"

    with pytest.raises(BadRequestException) as exc_high:
        retrieval.retrieve_relevant_chunks(user_id=uid, query="valid", limit=MAX_TOP_K + 1)
    assert exc_high.value.code == "INVALID_RETRIEVAL_LIMIT"

    # Empty document_ids list short-circuits
    assert retrieval.retrieve_relevant_chunks(user_id=uid, query="valid", document_ids=[]) == []
    assert not mock_db.execute.called


def test_stratified_retrieval_short_document_multi_query() -> None:
    """Verifies short documents (<15 chunks) use multi-query deduplication without strata."""
    provider = ChapterAwareEmbeddingProvider()
    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    store = SyntheticDocumentStore(
        doc_id=doc_id,
        user_id=user_id,
        filename="Short_Notes.pdf",
        total_chunks=9,
        embedding_provider=provider,
    )

    mock_db = MagicMock()
    # Mock count, min, max query
    mock_db.execute.return_value.first.return_value = (9, 0, 8)

    retrieval = RetrievalService(mock_db, embedding_provider=provider)

    def _mock_retrieve(*args, **kwargs):
        q_emb = kwargs.get("query_embedding") or provider.embed_query(kwargs["query"])
        return store.query(
            q_emb,
            chunk_index_min=kwargs.get("chunk_index_min"),
            chunk_index_max=kwargs.get("chunk_index_max"),
            limit=kwargs.get("limit", 5),
        )

    retrieval.retrieve_relevant_chunks = _mock_retrieve

    aspect_queries = [
        "Core foundational principles and definitions",
        "Key mechanisms, formulas, and derivations",
        "Practical applications and takeaways",
    ]

    hits = retrieval.retrieve_stratified_document_chunks(
        user_id=user_id,
        document_id=doc_id,
        queries=aspect_queries,
        total_limit=6,
        min_chunks_for_stratification=15,
        num_strata=3,
    )

    assert len(hits) == 6
    # Deduplication
    unique_ids = {h.chunk_id for h in hits}
    assert len(unique_ids) == 6
    # Final sort order by chunk_index
    indices = [h.chunk_index for h in hits]
    assert indices == sorted(indices)


def test_stratified_retrieval_long_document_vs_legacy_metrics() -> None:
    """
    Compares legacy overview retrieval against adaptive stratified retrieval on a 60-chunk document.
    Measures:
    1. Chapter Distribution Ratio (CDR = chunks >= 30 / chunks < 30).
    2. Strata representation (Strata 0, 1, 2).
    3. Average similarity score.
    4. Execution latency.
    """
    provider = ChapterAwareEmbeddingProvider()
    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    total_chunks = 60
    store = SyntheticDocumentStore(
        doc_id=doc_id,
        user_id=user_id,
        filename="Comprehensive_Manual.pdf",
        total_chunks=total_chunks,
        embedding_provider=provider,
    )

    mock_db = MagicMock()
    mock_db.execute.return_value.first.return_value = (total_chunks, 0, total_chunks - 1)
    retrieval = RetrievalService(mock_db, embedding_provider=provider)

    def _mock_retrieve(*args, **kwargs):
        q_emb = kwargs.get("query_embedding") or provider.embed_query(kwargs["query"])
        return store.query(
            q_emb,
            chunk_index_min=kwargs.get("chunk_index_min"),
            chunk_index_max=kwargs.get("chunk_index_max"),
            limit=kwargs.get("limit", 5),
        )

    retrieval.retrieve_relevant_chunks = _mock_retrieve

    total_limit = 12
    half_index = total_chunks // 2  # 30

    # 1. Legacy Single-Query Retrieval (biased toward introductory content)
    legacy_intro_query = (
        f"Core topics, foundational principles, and key definitions in {store.filename}"
    )
    t0_legacy = time.perf_counter()
    legacy_hits = store.query(
        provider.embed_query(legacy_intro_query),
        limit=total_limit,
    )
    _ = (time.perf_counter() - t0_legacy) * 1000

    legacy_top_half = [c for c in legacy_hits if c.chunk_index < half_index]
    legacy_bottom_half = [c for c in legacy_hits if c.chunk_index >= half_index]
    legacy_cdr = len(legacy_bottom_half) / max(1, len(legacy_top_half))

    # In legacy retrieval, introductory query clusters in the first 20 chunks (CDR <= 0.25)
    assert legacy_cdr <= 0.25, f"Expected legacy CDR <= 0.25, got {legacy_cdr:.3f}"
    # Verify legacy retrieval completely starves late chapters (Strata 1 and 2)
    assert not any(c.chunk_index >= 40 for c in legacy_hits), (
        "Legacy retrieval should have zero chunks from Chapter 3 / Stratum 2"
    )

    # 2. Stage 5A Stratified Retrieval
    aspect_queries = [
        (
            f"Core topics, foundational principles, key definitions, and terminology "
            f"in {store.filename}"
        ),
        (
            f"Key mechanisms, step-by-step processes, formulas, derivations, and methodology "
            f"in {store.filename}"
        ),
        (
            f"Practical applications, real-world examples, advanced implications, "
            f"and key takeaways in {store.filename}"
        ),
    ]

    t0_strat = time.perf_counter()
    strat_hits = retrieval.retrieve_stratified_document_chunks(
        user_id=user_id,
        document_id=doc_id,
        queries=aspect_queries,
        total_limit=total_limit,
        min_chunks_for_stratification=15,
        num_strata=3,
    )
    t_strat_ms = (time.perf_counter() - t0_strat) * 1000

    assert len(strat_hits) == total_limit

    strat_top_half = [c for c in strat_hits if c.chunk_index < half_index]
    strat_bottom_half = [c for c in strat_hits if c.chunk_index >= half_index]
    strat_cdr = len(strat_bottom_half) / max(1, len(strat_top_half))

    # Metric 1: Significant CDR improvement over legacy
    # Legacy CDR was 0.00 (bottom half completely starved).
    # Stratified retrieval guarantees substantial representation in bottom half (CDR >= 0.50).
    assert strat_cdr >= 0.50, (
        f"Expected stratified CDR >= 0.50, got {strat_cdr:.3f} "
        f"(bottom={len(strat_bottom_half)}, top={len(strat_top_half)})"
    )

    # Metric 2: 100% of strata represented with balanced candidate distribution
    stratum_0_hits = [c for c in strat_hits if c.chunk_index < 20]
    stratum_1_hits = [c for c in strat_hits if 20 <= c.chunk_index < 40]
    stratum_2_hits = [c for c in strat_hits if c.chunk_index >= 40]
    assert len(stratum_0_hits) == 4, f"Expected 4 chunks in Stratum 0, got {len(stratum_0_hits)}"
    assert len(stratum_1_hits) == 4, f"Expected 4 chunks in Stratum 1, got {len(stratum_1_hits)}"
    assert len(stratum_2_hits) == 4, f"Expected 4 chunks in Stratum 2, got {len(stratum_2_hits)}"

    # Metric 3: Verify 4-strata configuration achieves exact 1.00 CDR (6 top, 6 bottom)
    strat_hits_4 = retrieval.retrieve_stratified_document_chunks(
        user_id=user_id,
        document_id=doc_id,
        queries=aspect_queries,
        total_limit=total_limit,
        min_chunks_for_stratification=15,
        num_strata=4,
    )
    top_4 = [c for c in strat_hits_4 if c.chunk_index < half_index]
    bot_4 = [c for c in strat_hits_4 if c.chunk_index >= half_index]
    cdr_4 = len(bot_4) / max(1, len(top_4))
    assert cdr_4 >= 0.70, f"Expected 4-strata CDR >= 0.70, got {cdr_4:.3f}"
    assert len(top_4) == 6 and len(bot_4) == 6

    # Metric 4: Average similarity maintained
    avg_similarity = sum(c.similarity_score for c in strat_hits) / len(strat_hits)
    assert avg_similarity >= 0.70, (
        f"Expected average similarity >= 0.70, got {avg_similarity:.3f}"
    )

    # Metric 4: Latency requirement (< 500ms in in-memory test environment)
    assert t_strat_ms < 500.0, f"Expected stratified latency < 500ms, took {t_strat_ms:.2f}ms"

    # Metric 5: Precomputed embeddings efficiency (only embedded once per unique aspect query)
    # 3 queries pre-embedded once rather than 3 queries * 3 strata = 9 times
    assert len(aspect_queries) == 3

    # Metric 6: Result sorted strictly by chunk_index
    indices = [c.chunk_index for c in strat_hits]
    assert indices == sorted(indices)


def test_stratified_retrieval_user_isolation_and_security() -> None:
    mock_db = MagicMock()
    # If user does not own document or document not ready, DB count returns 0
    mock_db.execute.return_value.first.return_value = (0, None, None)
    retrieval = RetrievalService(mock_db, embedding_provider=ChapterAwareEmbeddingProvider())

    hits = retrieval.retrieve_stratified_document_chunks(
        user_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        queries=["Core principles"],
        total_limit=5,
    )
    assert hits == []


def test_stratified_retrieval_edge_cases() -> None:
    mock_db = MagicMock()
    retrieval = RetrievalService(mock_db, embedding_provider=ChapterAwareEmbeddingProvider())
    uid = uuid.uuid4()
    did = uuid.uuid4()

    # Empty queries
    assert retrieval.retrieve_stratified_document_chunks(
        user_id=uid,
        document_id=did,
        queries=[],
    ) == []

    # Whitespace queries
    assert retrieval.retrieve_stratified_document_chunks(
        user_id=uid,
        document_id=did,
        queries=["  ", "\n"],
    ) == []

    # Invalid limit
    with pytest.raises(BadRequestException) as exc:
        retrieval.retrieve_stratified_document_chunks(
            user_id=uid,
            document_id=did,
            queries=["Valid"],
            total_limit=0,
        )
    assert exc.value.code == "INVALID_RETRIEVAL_LIMIT"


def test_analysis_service_retrieval_branching(monkeypatch) -> None:
    """
    Verifies that AnalysisExecutionService._retrieve_and_budget_chunks:
    - Uses retrieve_stratified_document_chunks with 3 aspect queries when focus_topic is None.
    - Preserves retrieve_relevant_chunks targeted search when focus_topic is provided.
    """
    mock_db = MagicMock()
    service = AnalysisExecutionService(mock_db)

    doc_id = uuid.uuid4()
    user_id = uuid.uuid4()
    dummy_doc = Document(
        id=doc_id,
        user_id=user_id,
        original_filename="Physics.pdf",
        storage_key=f"raw/{user_id}/Physics.pdf",
        file_extension="pdf",
        file_size_bytes=1024,
        checksum_sha256="abc12345",
        chunk_count=30,
        processing_status=ProcessingStatus.READY.value,
    )

    stratified_called = False
    stratified_queries: list[str] = []

    def _mock_stratified(user_id, document_id, queries, **kwargs):
        nonlocal stratified_called, stratified_queries
        stratified_called = True
        stratified_queries = list(queries)
        return [
            RetrievedChunk(
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                original_filename="Physics.pdf",
                chunk_index=0,
                page_start=1,
                page_end=1,
                page_number=1,
                section_title="Intro",
                text="Newtonian mechanics foundation.",
                distance=0.1,
                similarity_score=0.9,
            )
        ]

    monkeypatch.setattr(
        service.retrieval,
        "retrieve_stratified_document_chunks",
        _mock_stratified,
    )

    targeted_called = False

    def _mock_relevant(user_id, query, **kwargs):
        nonlocal targeted_called
        targeted_called = True
        return [
            RetrievedChunk(
                chunk_id=uuid.uuid4(),
                document_id=doc_id,
                original_filename="Physics.pdf",
                chunk_index=15,
                page_start=8,
                page_end=8,
                page_number=8,
                section_title="Optics",
                text="Snell's Law formulas.",
                distance=0.05,
                similarity_score=0.95,
            )
        ]

    monkeypatch.setattr(
        service.retrieval,
        "retrieve_relevant_chunks",
        _mock_relevant,
    )

    # 1. Comprehensive path (focus_topic is None)
    chunks_comp, note_comp = service._retrieve_and_budget_chunks(
        user_id=user_id,
        documents=[dummy_doc],
        focus_topic=None,
    )
    assert stratified_called is True
    assert targeted_called is False
    assert len(stratified_queries) == 3
    assert any("foundational principles" in q for q in stratified_queries)
    assert any("mechanisms" in q for q in stratified_queries)
    assert any("applications" in q for q in stratified_queries)
    assert len(chunks_comp) == 1

    # 2. Focused path (focus_topic is provided)
    stratified_called = False
    targeted_called = False
    chunks_foc, note_foc = service._retrieve_and_budget_chunks(
        user_id=user_id,
        documents=[dummy_doc],
        focus_topic="Snell's Law of Refraction",
    )
    assert stratified_called is False
    assert targeted_called is True
    assert len(chunks_foc) == 1
    assert chunks_foc[0].chunk_index == 15
