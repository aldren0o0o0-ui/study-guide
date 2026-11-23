import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import BadRequestException
from app.database.vector import cosine_distance_expression, is_pgvector_enabled
from app.modules.documents.models import Document, DocumentChunk, ProcessingStatus
from app.modules.processing.embeddings import EmbeddingProvider, create_embedding_provider

logger = logging.getLogger("app.modules.processing.retrieval")

DEFAULT_TOP_K = 5
MAX_TOP_K = 50


@dataclass(frozen=True)
class RetrievedChunk:
    """
    A user-owned document chunk matched via vector similarity search,
    including complete source attribution metadata for grounded AI citations.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    original_filename: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    page_number: int | None
    section_title: str | None
    text: str
    distance: float
    similarity_score: float


class RetrievalService:
    """
    User-isolated vector similarity retrieval service over `DocumentChunk` and `Document`.

    Security invariant:
    Every retrieval query enforces `Document.user_id == user_id`,
    `DocumentChunk.user_id == user_id`, and `Document.processing_status == 'ready'`.
    """

    def __init__(
        self,
        db: Session,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or create_embedding_provider(self.settings)

    def retrieve_relevant_chunks(
        self,
        *,
        user_id: uuid.UUID,
        query: str,
        document_ids: Sequence[uuid.UUID] | None = None,
        limit: int = DEFAULT_TOP_K,
        chunk_index_min: int | None = None,
        chunk_index_max: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise BadRequestException(
                "Search query cannot be empty",
                code="EMPTY_RETRIEVAL_QUERY",
            )

        if limit < 1 or limit > MAX_TOP_K:
            raise BadRequestException(
                f"Retrieval limit must be between 1 and {MAX_TOP_K}",
                code="INVALID_RETRIEVAL_LIMIT",
            )

        if document_ids is not None and len(document_ids) == 0:
            return []

        if query_embedding is None:
            query_embedding = self.embedding_provider.embed_query(cleaned_query)

        use_pgvector = is_pgvector_enabled(self.db)
        distance_expr = cosine_distance_expression(
            DocumentChunk.embedding,
            query_embedding,
            use_pgvector=use_pgvector,
        ).label("distance")

        stmt = (
            select(
                DocumentChunk,
                Document.original_filename,
                distance_expr,
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.user_id == user_id,
                DocumentChunk.user_id == user_id,
                Document.processing_status == ProcessingStatus.READY.value,
            )
        )

        if document_ids is not None:
            unique_doc_ids = list(dict.fromkeys(document_ids))
            stmt = stmt.where(Document.id.in_(unique_doc_ids))

        if chunk_index_min is not None:
            stmt = stmt.where(DocumentChunk.chunk_index >= chunk_index_min)
        if chunk_index_max is not None:
            stmt = stmt.where(DocumentChunk.chunk_index <= chunk_index_max)

        stmt = stmt.order_by(distance_expr.asc(), DocumentChunk.chunk_index.asc()).limit(limit)

        rows = self.db.execute(stmt).all()

        results: list[RetrievedChunk] = []
        for chunk, original_filename, raw_distance in rows:
            dist = float(raw_distance) if raw_distance is not None else 1.0
            similarity = round(max(0.0, min(1.0, 1.0 - dist)), 6)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    original_filename=str(original_filename),
                    chunk_index=chunk.chunk_index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    page_number=chunk.page_start,
                    section_title=chunk.section_title,
                    text=chunk.content,
                    distance=round(dist, 6),
                    similarity_score=similarity,
                )
            )

        return results

    def retrieve_stratified_document_chunks(
        self,
        *,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        queries: Sequence[str],
        total_limit: int = DEFAULT_TOP_K,
        min_chunks_for_stratification: int = 15,
        num_strata: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Adaptive multi-query & chapter-strided retrieval for a single document.

        Mitigates introductory bias on long-form materials (>30-50 pages / >= 15 chunks)
        by partitioning the document's chunk index span into `num_strata` non-overlapping strata
        (Beginning, Middle, End), running the specified pedagogical queries per stratum,
        and ensuring balanced chapter representation across the entire document.

        Short documents (< min_chunks_for_stratification) use adaptive multi-query without
        strata partitioning.
        """
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return []

        if total_limit < 1:
            raise BadRequestException(
                "Retrieval limit must be at least 1",
                code="INVALID_RETRIEVAL_LIMIT",
            )
        total_limit = min(total_limit, MAX_TOP_K)

        # Count total ready chunks and find index span for this document
        stmt = (
            select(
                func.count(DocumentChunk.id),
                func.min(DocumentChunk.chunk_index),
                func.max(DocumentChunk.chunk_index),
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
                DocumentChunk.user_id == user_id,
                Document.processing_status == ProcessingStatus.READY.value,
            )
        )
        row = self.db.execute(stmt).first()
        chunk_count = row[0] if row else 0
        if not chunk_count or chunk_count == 0:
            return []

        min_idx = row[1] if row[1] is not None else 0
        max_idx = row[2] if row[2] is not None else (chunk_count - 1)

        # Precompute query embeddings once for all queries
        query_embeddings = [
            (q, self.embedding_provider.embed_query(q))
            for q in clean_queries
        ]

        # Case 1: Short document or single stratum -> Multi-query over entire document
        if chunk_count < min_chunks_for_stratification or num_strata <= 1:
            candidate_map: dict[uuid.UUID, RetrievedChunk] = {}
            for q, q_emb in query_embeddings:
                hits = self.retrieve_relevant_chunks(
                    user_id=user_id,
                    query=q,
                    document_ids=[document_id],
                    limit=total_limit,
                    query_embedding=q_emb,
                )
                for hit in hits:
                    prev = candidate_map.get(hit.chunk_id)
                    if prev is None or hit.similarity_score > prev.similarity_score:
                        candidate_map[hit.chunk_id] = hit

            # Pick top candidates by similarity, then sort in chunk_index order
            sorted_candidates = sorted(
                candidate_map.values(),
                key=lambda c: c.similarity_score,
                reverse=True,
            )
            selected = sorted_candidates[:total_limit]
            selected.sort(key=lambda c: c.chunk_index)
            return selected

        # Case 2: Long document -> Stratified chapter retrieval across num_strata
        span = max_idx - min_idx + 1
        effective_num_strata = max(2, min(num_strata, 10))
        target_per_stratum = max(1, total_limit // effective_num_strata)

        strata_selected: list[RetrievedChunk] = []
        seen_chunk_ids: set[uuid.UUID] = set()
        all_pool_candidates: dict[uuid.UUID, RetrievedChunk] = {}

        for s in range(effective_num_strata):
            stratum_start = min_idx + int(s * span / effective_num_strata)
            if s == effective_num_strata - 1:
                stratum_end = max_idx
            else:
                stratum_end = min_idx + int((s + 1) * span / effective_num_strata) - 1

            stratum_candidates: dict[uuid.UUID, RetrievedChunk] = {}
            for q, q_emb in query_embeddings:
                hits = self.retrieve_relevant_chunks(
                    user_id=user_id,
                    query=q,
                    document_ids=[document_id],
                    limit=target_per_stratum * 2,
                    chunk_index_min=stratum_start,
                    chunk_index_max=stratum_end,
                    query_embedding=q_emb,
                )
                for hit in hits:
                    prev = stratum_candidates.get(hit.chunk_id)
                    if prev is None or hit.similarity_score > prev.similarity_score:
                        stratum_candidates[hit.chunk_id] = hit
                    prev_pool = all_pool_candidates.get(hit.chunk_id)
                    if prev_pool is None or hit.similarity_score > prev_pool.similarity_score:
                        all_pool_candidates[hit.chunk_id] = hit

            # Sort stratum hits by similarity descending and take target_per_stratum
            sorted_stratum = sorted(
                stratum_candidates.values(),
                key=lambda c: c.similarity_score,
                reverse=True,
            )
            stratum_picks = [c for c in sorted_stratum if c.chunk_id not in seen_chunk_ids][
                :target_per_stratum
            ]
            for pick in stratum_picks:
                seen_chunk_ids.add(pick.chunk_id)
                strata_selected.append(pick)

        # Backfill remaining budget up to total_limit from remaining candidates
        if len(strata_selected) < total_limit:
            remaining_candidates = [
                c for c in all_pool_candidates.values() if c.chunk_id not in seen_chunk_ids
            ]
            remaining_candidates.sort(key=lambda c: c.similarity_score, reverse=True)
            needed = total_limit - len(strata_selected)
            for pick in remaining_candidates[:needed]:
                seen_chunk_ids.add(pick.chunk_id)
                strata_selected.append(pick)

        # Return final selected chunks sorted by chunk_index for pedagogical narrative flow
        strata_selected.sort(key=lambda c: c.chunk_index)
        return strata_selected
