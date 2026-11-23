import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app.modules.processing.retrieval import RetrievedChunk
from app.modules.voice_practice.models import (
    VoiceEvaluationStatus,
    VoicePracticeFeedback,
    VoicePracticeSession,
    VoicePracticeSessionChunk,
    VoicePracticeSessionDocument,
    VoicePracticeTurn,
    VoiceSessionStatus,
)


class VoicePracticeRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_session(
        self,
        *,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        mode: str,
        title: str,
        config_json: dict[str, Any],
        document_ids: Sequence[uuid.UUID] | None = None,
        retrieved_chunks: Sequence[tuple[str, RetrievedChunk]] | None = None,
    ) -> VoicePracticeSession:
        now = datetime.now(UTC)
        session = VoicePracticeSession(
            id=session_id,
            user_id=user_id,
            mode=mode,
            title=title[:255],
            status=VoiceSessionStatus.PENDING.value,
            evaluation_status=VoiceEvaluationStatus.NOT_STARTED.value,
            config_json=config_json,
            duration_seconds=0.0,
            created_at=now,
            updated_at=now,
        )
        self.db.add(session)
        self.db.flush()

        if document_ids:
            seen_docs: set[uuid.UUID] = set()
            for pos, doc_id in enumerate(document_ids):
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                self.db.add(
                    VoicePracticeSessionDocument(
                        session_id=session.id,
                        document_id=doc_id,
                        position=pos,
                        created_at=now,
                    )
                )

        if retrieved_chunks:
            seen_chunks: set[uuid.UUID] = set()
            for pos, (source_key, chunk) in enumerate(retrieved_chunks):
                if chunk.chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk.chunk_id)
                self.db.add(
                    VoicePracticeSessionChunk(
                        session_id=session.id,
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        source_key=source_key,
                        position=pos,
                        similarity_score=chunk.similarity_score,
                    )
                )

        self.db.flush()
        return session

    def mark_session_active(
        self,
        session_id: uuid.UUID,
        *,
        started_at: datetime | None = None,
    ) -> VoicePracticeSession | None:
        session = self.db.get(VoicePracticeSession, session_id)
        if session is None:
            return None
        now = started_at or datetime.now(UTC)
        session.status = VoiceSessionStatus.ACTIVE.value
        if session.started_at is None:
            session.started_at = now
        session.updated_at = now
        self.db.flush()
        return session

    def save_turn(
        self,
        *,
        session_id: uuid.UUID,
        turn_index: int,
        speaker_role: str,
        transcript_text: str,
        turn_status: str,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> VoicePracticeTurn:
        """
        Idempotently persists a finalized turn. If a turn with `(session_id, turn_index)`
        already exists, updates it in-place rather than inserting a duplicate row.
        """
        cleaned_text = transcript_text.strip()
        now = datetime.now(UTC)
        stmt = select(VoicePracticeTurn).where(
            VoicePracticeTurn.session_id == session_id,
            VoicePracticeTurn.turn_index == turn_index,
        )
        existing = self.db.scalars(stmt).first()
        if existing is not None:
            existing.speaker_role = speaker_role
            existing.transcript_text = cleaned_text
            existing.turn_status = turn_status
            existing.completed_at = completed_at or now
            if metadata_json is not None:
                existing.metadata_json = metadata_json
            self.db.flush()
            return existing

        turn = VoicePracticeTurn(
            session_id=session_id,
            turn_index=turn_index,
            speaker_role=speaker_role,
            transcript_text=cleaned_text,
            turn_status=turn_status,
            started_at=started_at or now,
            completed_at=completed_at or now,
            metadata_json=metadata_json,
        )
        self.db.add(turn)

        session = self.db.get(VoicePracticeSession, session_id)
        if session is not None:
            session.updated_at = now

        self.db.flush()
        return turn

    def finalize_session(
        self,
        session_id: uuid.UUID,
        *,
        status: str,
        termination_reason: str,
        duration_seconds: float,
        ended_at: datetime | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> VoicePracticeSession | None:
        session = self.db.get(VoicePracticeSession, session_id)
        if session is None:
            return None

        now = ended_at or datetime.now(UTC)
        user_turns_stmt = (
            select(func.count())
            .select_from(VoicePracticeTurn)
            .where(
                VoicePracticeTurn.session_id == session_id,
                VoicePracticeTurn.speaker_role == "user",
            )
        )
        user_turns_count = int(self.db.scalar(user_turns_stmt) or 0)

        session.status = status
        session.termination_reason = termination_reason
        session.duration_seconds = max(0.0, round(duration_seconds, 2))
        session.ended_at = now
        session.updated_at = now
        session.error_code = error_code
        session.error_message = error_message[:500] if error_message else None

        if session.evaluation_status == VoiceEvaluationStatus.NOT_STARTED.value:
            if user_turns_count > 0:
                session.evaluation_status = VoiceEvaluationStatus.PENDING.value
            else:
                session.evaluation_status = VoiceEvaluationStatus.NOT_STARTED.value

        self.db.flush()
        return session

    def get_session_for_user(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> VoicePracticeSession | None:
        stmt = (
            select(VoicePracticeSession)
            .options(
                selectinload(VoicePracticeSession.turns),
                selectinload(VoicePracticeSession.session_documents).selectinload(
                    VoicePracticeSessionDocument.document
                ),
                selectinload(VoicePracticeSession.session_chunks).selectinload(
                    VoicePracticeSessionChunk.document
                ),
                selectinload(VoicePracticeSession.session_chunks).selectinload(
                    VoicePracticeSessionChunk.chunk
                ),
                selectinload(VoicePracticeSession.feedback),
            )
            .where(
                VoicePracticeSession.id == session_id,
                VoicePracticeSession.user_id == user_id,
            )
        )
        return self.db.scalars(stmt).first()

    def list_sessions_for_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
        mode: str | None = None,
        status: str | None = None,
    ) -> tuple[list[VoicePracticeSession], int]:
        filters = [VoicePracticeSession.user_id == user_id]
        if mode:
            filters.append(VoicePracticeSession.mode == mode)
        if status:
            filters.append(VoicePracticeSession.status == status)

        count_stmt = select(func.count()).select_from(VoicePracticeSession).where(*filters)
        total = int(self.db.scalar(count_stmt) or 0)

        items_stmt = (
            select(VoicePracticeSession)
            .options(
                selectinload(VoicePracticeSession.turns),
                selectinload(VoicePracticeSession.session_documents),
                selectinload(VoicePracticeSession.feedback),
            )
            .where(*filters)
            .order_by(VoicePracticeSession.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list(self.db.scalars(items_stmt).all())
        return items, total

    def list_turns_for_session(
        self,
        session_id: uuid.UUID,
    ) -> list[VoicePracticeTurn]:
        stmt = (
            select(VoicePracticeTurn)
            .where(VoicePracticeTurn.session_id == session_id)
            .order_by(VoicePracticeTurn.turn_index.asc())
        )
        return list(self.db.scalars(stmt).all())

    def get_feedback_for_session(
        self,
        session_id: uuid.UUID,
    ) -> VoicePracticeFeedback | None:
        stmt = select(VoicePracticeFeedback).where(VoicePracticeFeedback.session_id == session_id)
        return self.db.scalars(stmt).first()

    def reconcile_expired_pending_sessions(
        self,
        user_id: uuid.UUID | None = None,
        *,
        ttl_seconds: int = 60,
        now: datetime | None = None,
    ) -> int:
        """
        Transitions stale `pending` sessions whose connection ticket TTL has expired
        into `failed` (`termination_reason='ticket_expired'`) so abandoned tickets never
        leave permanent pending sessions in history.
        """
        ts = now or datetime.now(UTC)
        cutoff = ts - timedelta(seconds=max(1, ttl_seconds))
        filters = [
            VoicePracticeSession.status == VoiceSessionStatus.PENDING.value,
            VoicePracticeSession.created_at <= cutoff,
        ]
        if user_id is not None:
            filters.append(VoicePracticeSession.user_id == user_id)

        stmt = (
            update(VoicePracticeSession)
            .where(*filters)
            .values(
                status=VoiceSessionStatus.FAILED.value,
                termination_reason="ticket_expired",
                error_code="TICKET_EXPIRED",
                error_message="Connection ticket expired before WebSocket session started.",
                ended_at=ts,
                updated_at=ts,
            )
        )
        result = self.db.execute(stmt)
        self.db.flush()
        return int(getattr(result, "rowcount", 0) or 0)

    def claim_session_for_evaluation(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        model_name: str,
    ) -> bool:
        """
        Atomically claims a session for evaluation in PostgreSQL so concurrent requests
        cannot evaluate the same session simultaneously.
        """
        now = datetime.now(UTC)
        stmt = (
            update(VoicePracticeSession)
            .where(
                VoicePracticeSession.id == session_id,
                VoicePracticeSession.user_id == user_id,
                VoicePracticeSession.status.notin_(
                    [VoiceSessionStatus.PENDING.value, VoiceSessionStatus.ACTIVE.value]
                ),
                VoicePracticeSession.evaluation_status.notin_(
                    [
                        VoiceEvaluationStatus.PROCESSING.value,
                        VoiceEvaluationStatus.COMPLETED.value,
                    ]
                ),
            )
            .values(
                evaluation_status=VoiceEvaluationStatus.PROCESSING.value,
                updated_at=now,
            )
        )
        result = self.db.execute(stmt)
        if int(getattr(result, "rowcount", 0) or 0) == 0:
            return False

        feedback = self.get_feedback_for_session(session_id)
        if feedback is None:
            feedback = VoicePracticeFeedback(
                session_id=session_id,
                evaluation_status=VoiceEvaluationStatus.PROCESSING.value,
                model_name=model_name,
                created_at=now,
                updated_at=now,
            )
            self.db.add(feedback)
        else:
            feedback.evaluation_status = VoiceEvaluationStatus.PROCESSING.value
            feedback.model_name = model_name
            feedback.error_code = None
            feedback.error_message = None
            feedback.updated_at = now

        self.db.flush()
        return True

    def upsert_feedback_processing(
        self,
        session_id: uuid.UUID,
        *,
        model_name: str,
    ) -> VoicePracticeFeedback:
        now = datetime.now(UTC)
        session = self.db.get(VoicePracticeSession, session_id)
        if session is not None:
            session.evaluation_status = VoiceEvaluationStatus.PROCESSING.value
            session.updated_at = now

        feedback = self.get_feedback_for_session(session_id)
        if feedback is None:
            feedback = VoicePracticeFeedback(
                session_id=session_id,
                evaluation_status=VoiceEvaluationStatus.PROCESSING.value,
                model_name=model_name,
                created_at=now,
                updated_at=now,
            )
            self.db.add(feedback)
        else:
            feedback.evaluation_status = VoiceEvaluationStatus.PROCESSING.value
            feedback.model_name = model_name
            feedback.error_code = None
            feedback.error_message = None
            feedback.updated_at = now

        self.db.flush()
        return feedback

    def save_feedback_completed(
        self,
        session_id: uuid.UUID,
        *,
        model_name: str,
        overall_score: float,
        dimension_scores: list[dict[str, Any]],
        strengths: list[str],
        areas_for_improvement: list[str],
        recommendations: list[str],
        structured_feedback: dict[str, Any],
    ) -> VoicePracticeFeedback:
        now = datetime.now(UTC)
        session = self.db.get(VoicePracticeSession, session_id)
        if session is not None:
            session.evaluation_status = VoiceEvaluationStatus.COMPLETED.value
            session.updated_at = now

        feedback = self.get_feedback_for_session(session_id)
        if feedback is None:
            feedback = VoicePracticeFeedback(
                session_id=session_id,
                created_at=now,
            )
            self.db.add(feedback)

        feedback.evaluation_status = VoiceEvaluationStatus.COMPLETED.value
        feedback.overall_score = round(overall_score, 2)
        feedback.dimension_scores_json = dimension_scores
        feedback.strengths_json = strengths
        feedback.areas_for_improvement_json = areas_for_improvement
        feedback.recommendations_json = recommendations
        feedback.structured_feedback_json = structured_feedback
        feedback.model_name = model_name
        feedback.error_code = None
        feedback.error_message = None
        feedback.updated_at = now

        self.db.flush()
        return feedback

    def save_feedback_failed(
        self,
        session_id: uuid.UUID,
        *,
        model_name: str,
        error_code: str,
        error_message: str,
    ) -> VoicePracticeFeedback:
        now = datetime.now(UTC)
        session = self.db.get(VoicePracticeSession, session_id)
        if session is not None:
            session.evaluation_status = VoiceEvaluationStatus.FAILED.value
            session.updated_at = now

        feedback = self.get_feedback_for_session(session_id)
        if feedback is None:
            feedback = VoicePracticeFeedback(
                session_id=session_id,
                created_at=now,
            )
            self.db.add(feedback)

        feedback.evaluation_status = VoiceEvaluationStatus.FAILED.value
        feedback.overall_score = None
        feedback.model_name = model_name
        feedback.error_code = error_code[:64]
        feedback.error_message = error_message[:500]
        feedback.updated_at = now

        self.db.flush()
        return feedback

    def delete_session(self, session: VoicePracticeSession) -> None:
        self.db.delete(session)
        self.db.flush()
