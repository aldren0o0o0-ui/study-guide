import hashlib
import json
import logging
import secrets
import threading
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import redis
from fastapi import status
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AppException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    NotFoundException,
    ServiceUnavailableException,
    UnauthorizedException,
)
from app.database.session import get_session_factory
from app.modules.auth.models import AccountStatus, User
from app.modules.auth.repository import AuthRepository
from app.modules.documents.models import Document, ProcessingStatus
from app.modules.processing.retrieval import RetrievalService, RetrievedChunk
from app.modules.voice_practice.evaluator import (
    TurnTranscriptInput,
    VoiceEvaluationError,
    VoicePracticeEvaluator,
    assign_voice_source_keys,
    build_live_system_instruction,
    create_voice_practice_evaluator,
)
from app.modules.voice_practice.models import (
    VoiceEvaluationStatus,
    VoicePracticeFeedback,
    VoicePracticeSession,
    VoicePracticeTurn,
    VoiceSessionStatus,
)
from app.modules.voice_practice.repository import VoicePracticeRepository
from app.modules.voice_practice.schemas import (
    ActiveVoiceSession,
    CommunicationPracticeConfig,
    DocumentOralPracticeConfig,
    InterviewPracticeConfig,
    StoredSessionTicket,
    VoiceAudioFormatSpec,
    VoiceDimensionScore,
    VoiceGroundingStatus,
    VoicePracticeDocumentRefRead,
    VoicePracticeFeedbackRead,
    VoicePracticeMode,
    VoicePracticeSessionDeleteResponse,
    VoicePracticeSessionDetailRead,
    VoicePracticeSessionListResponse,
    VoicePracticeSessionSummaryRead,
    VoicePracticeSourceReferenceRead,
    VoicePracticeTurnRead,
    VoiceSessionLimitsSpec,
    VoiceSessionTicketRequest,
    VoiceSessionTicketResponse,
    VoiceTranscriptEvidence,
)

logger = logging.getLogger("app.modules.voice_practice.service")


class TooManyRequestsException(AppException):
    def __init__(
        self,
        message: str = "Rate limit or quota exceeded",
        *,
        code: str = "QUOTA_EXCEEDED",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details,
        )


def generate_connection_ticket() -> str:
    """Generates a cryptographically secure 256-bit URL-safe single-use ticket."""
    return secrets.token_urlsafe(32)


def hash_connection_ticket(raw_ticket: str) -> str:
    """Hashes a raw connection ticket with SHA-256 so raw secrets are never stored or logged."""
    return hashlib.sha256(raw_ticket.encode("utf-8")).hexdigest()


def current_utc_date_key(now: datetime | None = None) -> str:
    """Returns the canonical UTC date string (YYYY-MM-DD) following project timezone conventions."""
    ts = now or datetime.now(UTC)
    return ts.astimezone(UTC).date().isoformat()


def _build_session_title(
    payload: VoiceSessionTicketRequest,
    documents: Sequence[Document] | None = None,
) -> str:
    if payload.title and payload.title.strip():
        return payload.title.strip()[:255]

    if payload.mode == VoicePracticeMode.INTERVIEW:
        icfg = payload.interview_config or InterviewPracticeConfig()
        return f"Interview Practice: {icfg.target_role} ({icfg.interview_type.value})"[:255]

    if payload.mode == VoicePracticeMode.DOCUMENT_ORAL:
        dcfg = payload.document_oral_config or DocumentOralPracticeConfig()
        if dcfg.focus_topic and dcfg.focus_topic.strip():
            return f"Oral Practice: {dcfg.focus_topic.strip()}"[:255]
        if documents:
            if len(documents) == 1:
                return f"Oral Practice: {documents[0].original_filename}"[:255]
            return (f"Oral Practice: {documents[0].original_filename} + {len(documents) - 1} more")[
                :255
            ]
        return "Document-Based Oral Practice"

    if payload.mode == VoicePracticeMode.COMMUNICATION:
        ccfg = payload.communication_config or CommunicationPracticeConfig()
        if ccfg.speaking_topic and ccfg.speaking_topic.strip():
            return f"Communication: {ccfg.speaking_topic.strip()}"[:255]
        scenario_label = ccfg.scenario.value.replace("_", " ").title()
        return f"Communication Practice: {scenario_label}"[:255]

    return "Voice Practice Session"


def _build_session_config_dict(payload: VoiceSessionTicketRequest) -> dict[str, Any]:
    config: dict[str, Any] = {
        "mode": payload.mode.value,
        "input_sample_rate_hz": payload.input_sample_rate_hz,
    }
    if payload.mode == VoicePracticeMode.INTERVIEW and payload.interview_config:
        config["interview_config"] = payload.interview_config.model_dump(mode="json")
    elif payload.mode in (VoicePracticeMode.COMMUNICATION, VoicePracticeMode.NEUTRAL):
        if payload.communication_config:
            config["communication_config"] = payload.communication_config.model_dump(mode="json")
    elif payload.mode == VoicePracticeMode.DOCUMENT_ORAL and payload.document_oral_config:
        config["document_oral_config"] = payload.document_oral_config.model_dump(mode="json")
    return config


def _to_turn_read(turn: VoicePracticeTurn) -> VoicePracticeTurnRead:
    role: Literal["user", "assistant"] = "user" if turn.speaker_role == "user" else "assistant"
    t_status: Literal["completed", "interrupted"] = (
        "interrupted" if turn.turn_status == "interrupted" else "completed"
    )
    return VoicePracticeTurnRead(
        id=turn.id,
        session_id=turn.session_id,
        turn_index=turn.turn_index,
        speaker_role=role,
        transcript_text=turn.transcript_text,
        turn_status=t_status,
        started_at=turn.started_at,
        completed_at=turn.completed_at,
        metadata=turn.metadata_json,
    )


def _to_feedback_read(feedback: VoicePracticeFeedback) -> VoicePracticeFeedbackRead:
    structured = feedback.structured_feedback_json or {}
    raw_dims = feedback.dimension_scores_json or []
    dims = [VoiceDimensionScore.model_validate(d) for d in raw_dims]

    raw_evidence = structured.get("transcript_evidence") or []
    evidence = [VoiceTranscriptEvidence.model_validate(e) for e in raw_evidence]

    raw_sources = structured.get("source_references") or []
    sources = [VoicePracticeSourceReferenceRead.model_validate(s) for s in raw_sources]

    return VoicePracticeFeedbackRead(
        id=feedback.id,
        session_id=feedback.session_id,
        evaluation_status=feedback.evaluation_status,
        overall_score=feedback.overall_score,
        summary=structured.get("summary"),
        dimension_scores=dims,
        strengths=list(feedback.strengths_json or []),
        areas_for_improvement=list(feedback.areas_for_improvement_json or []),
        recommendations=list(feedback.recommendations_json or []),
        transcript_evidence=evidence,
        missing_concepts=list(structured.get("missing_concepts") or []),
        unsupported_statements=list(structured.get("unsupported_statements") or []),
        source_references=sources,
        model_name=feedback.model_name,
        error_code=feedback.error_code,
        error_message=feedback.error_message,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


def _resolve_grounding_status(session: VoicePracticeSession) -> VoiceGroundingStatus:
    cfg = session.config_json or {}
    raw = cfg.get("grounding_status")
    if isinstance(raw, str):
        try:
            return VoiceGroundingStatus(raw)
        except ValueError:
            pass
    if session.mode == VoicePracticeMode.DOCUMENT_ORAL.value:
        if session.session_chunks:
            return VoiceGroundingStatus.GROUNDED
        return VoiceGroundingStatus.EMPTY_CONTEXT
    return VoiceGroundingStatus.NOT_APPLICABLE


def _to_session_summary_read(session: VoicePracticeSession) -> VoicePracticeSessionSummaryRead:
    overall_score = session.feedback.overall_score if session.feedback is not None else None
    return VoicePracticeSessionSummaryRead(
        id=session.id,
        mode=session.mode,
        title=session.title,
        status=session.status,
        evaluation_status=session.evaluation_status,
        grounding_status=_resolve_grounding_status(session),
        started_at=session.started_at,
        ended_at=session.ended_at,
        duration_seconds=session.duration_seconds,
        turns_count=len(session.turns or []),
        documents_count=len(session.session_documents or []),
        overall_score=overall_score,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _to_session_detail_read(session: VoicePracticeSession) -> VoicePracticeSessionDetailRead:
    doc_refs: list[VoicePracticeDocumentRefRead] = []
    for link in session.session_documents or []:
        if link.document is not None:
            doc_refs.append(
                VoicePracticeDocumentRefRead(
                    document_id=link.document_id,
                    original_filename=link.document.original_filename,
                    file_extension=link.document.file_extension,
                    processing_status=link.document.processing_status,
                )
            )
    turns_read = [_to_turn_read(t) for t in (session.turns or [])]
    feedback_read = _to_feedback_read(session.feedback) if session.feedback is not None else None
    return VoicePracticeSessionDetailRead(
        id=session.id,
        mode=session.mode,
        title=session.title,
        status=session.status,
        evaluation_status=session.evaluation_status,
        grounding_status=_resolve_grounding_status(session),
        configuration=session.config_json or {},
        termination_reason=session.termination_reason,
        error_code=session.error_code,
        error_message=session.error_message,
        started_at=session.started_at,
        ended_at=session.ended_at,
        duration_seconds=session.duration_seconds,
        source_documents=doc_refs,
        turns=turns_read,
        feedback=feedback_read,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


class VoiceSessionStore(Protocol):
    """Abstraction for ticket, concurrency, and daily quota state management."""

    def save_ticket(
        self,
        ticket_hash: str,
        ticket: StoredSessionTicket,
        ttl_seconds: int,
    ) -> None: ...

    def consume_ticket(
        self,
        ticket_hash: str,
        *,
        now: datetime | None = None,
    ) -> StoredSessionTicket | None: ...

    def is_user_session_active(self, user_id: uuid.UUID) -> bool: ...

    def get_global_active_count(self) -> int: ...

    def get_daily_session_count(
        self,
        user_id: uuid.UUID,
        *,
        date_key: str | None = None,
    ) -> int: ...

    def acquire_session(
        self,
        session: ActiveVoiceSession,
        *,
        max_user_concurrent: int,
        max_global_concurrent: int,
        max_daily_sessions: int,
        lock_ttl_seconds: int,
        date_key: str | None = None,
    ) -> None: ...

    def release_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        refund_daily_quota: bool = False,
        date_key: str | None = None,
    ) -> bool: ...


class InMemoryVoiceSessionStore:
    """
    Development and test fallback store for single-process execution when Redis is unavailable.

    NOTE: This fallback is thread-safe and asyncio-safe within a single Python worker process,
    but does NOT provide multi-worker distributed coordination across separate OS processes.
    """

    def __init__(self, *, now_fn: Callable[[], datetime] | None = None) -> None:
        self._lock = threading.Lock()
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._tickets: dict[str, StoredSessionTicket] = {}
        self._user_sessions: dict[
            uuid.UUID,
            dict[uuid.UUID, tuple[ActiveVoiceSession, datetime]],
        ] = {}
        self._global_sessions: dict[uuid.UUID, tuple[uuid.UUID, datetime]] = {}
        self._daily_counts: dict[tuple[uuid.UUID, str], int] = {}

    def _prune_expired_locked(self, now: datetime) -> None:
        expired_tickets = [k for k, v in self._tickets.items() if v.expires_at <= now]
        for k in expired_tickets:
            self._tickets.pop(k, None)

        expired_sessions = [sid for sid, (_uid, exp) in self._global_sessions.items() if exp <= now]
        for sid in expired_sessions:
            entry = self._global_sessions.pop(sid, None)
            if entry is not None:
                uid, _ = entry
                user_map = self._user_sessions.get(uid)
                if user_map:
                    user_map.pop(sid, None)
                    if not user_map:
                        self._user_sessions.pop(uid, None)

    def save_ticket(
        self,
        ticket_hash: str,
        ticket: StoredSessionTicket,
        ttl_seconds: int,
    ) -> None:
        _ = ttl_seconds
        with self._lock:
            now = self._now_fn()
            self._prune_expired_locked(now)
            self._tickets[ticket_hash] = ticket

    def consume_ticket(
        self,
        ticket_hash: str,
        *,
        now: datetime | None = None,
    ) -> StoredSessionTicket | None:
        with self._lock:
            effective_now = now or self._now_fn()
            self._prune_expired_locked(effective_now)
            ticket = self._tickets.pop(ticket_hash, None)
            if ticket is None or ticket.expires_at <= effective_now:
                return None
            return ticket

    def is_user_session_active(self, user_id: uuid.UUID) -> bool:
        with self._lock:
            self._prune_expired_locked(self._now_fn())
            user_map = self._user_sessions.get(user_id)
            return bool(user_map)

    def get_global_active_count(self) -> int:
        with self._lock:
            self._prune_expired_locked(self._now_fn())
            return len(self._global_sessions)

    def get_daily_session_count(
        self,
        user_id: uuid.UUID,
        *,
        date_key: str | None = None,
    ) -> int:
        with self._lock:
            d_key = date_key or current_utc_date_key(self._now_fn())
            return self._daily_counts.get((user_id, d_key), 0)

    def acquire_session(
        self,
        session: ActiveVoiceSession,
        *,
        max_user_concurrent: int,
        max_global_concurrent: int,
        max_daily_sessions: int,
        lock_ttl_seconds: int,
        date_key: str | None = None,
    ) -> None:
        with self._lock:
            now = self._now_fn()
            self._prune_expired_locked(now)
            d_key = date_key or current_utc_date_key(now)

            user_map = self._user_sessions.get(session.user_id, {})
            if len(user_map) >= max_user_concurrent:
                raise ConflictException(
                    "You already have an active Voice Practice session.",
                    code="VOICE_SESSION_ALREADY_ACTIVE",
                )

            if len(self._global_sessions) >= max_global_concurrent:
                raise ServiceUnavailableException(
                    "All voice practice channels are currently in use. Please try again shortly.",
                    code="VOICE_GLOBAL_CAPACITY_REACHED",
                )

            used_today = self._daily_counts.get((session.user_id, d_key), 0)
            if used_today >= max_daily_sessions:
                raise TooManyRequestsException(
                    f"Daily voice practice limit reached ({max_daily_sessions} sessions per day).",
                    code="VOICE_DAILY_QUOTA_EXCEEDED",
                    details={"daily_limit": max_daily_sessions, "used_today": used_today},
                )

            expires_at = now + timedelta(seconds=max(1, lock_ttl_seconds))
            if session.user_id not in self._user_sessions:
                self._user_sessions[session.user_id] = {}
            self._user_sessions[session.user_id][session.session_id] = (session, expires_at)
            self._global_sessions[session.session_id] = (session.user_id, expires_at)
            self._daily_counts[(session.user_id, d_key)] = used_today + 1

    def release_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        refund_daily_quota: bool = False,
        date_key: str | None = None,
    ) -> bool:
        with self._lock:
            now = self._now_fn()
            d_key = date_key or current_utc_date_key(now)
            removed = self._global_sessions.pop(session_id, None) is not None
            user_map = self._user_sessions.get(user_id)
            if user_map and session_id in user_map:
                user_map.pop(session_id, None)
                removed = True
                if not user_map:
                    self._user_sessions.pop(user_id, None)

            if removed and refund_daily_quota:
                current = self._daily_counts.get((user_id, d_key), 0)
                if current > 0:
                    self._daily_counts[(user_id, d_key)] = current - 1
            return removed

    def clear(self) -> None:
        with self._lock:
            self._tickets.clear()
            self._user_sessions.clear()
            self._global_sessions.clear()
            self._daily_counts.clear()


class RedisVoiceSessionStore:
    """
    Distributed Redis-backed store for tickets, per-user/global concurrency locks, and daily quotas.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    @staticmethod
    def _ticket_key(ticket_hash: str) -> str:
        return f"voice:ticket:{ticket_hash}"

    @staticmethod
    def _user_active_key(user_id: uuid.UUID) -> str:
        return f"voice:active:user:{user_id}"

    @staticmethod
    def _session_lock_key(session_id: uuid.UUID) -> str:
        return f"voice:active:session:{session_id}"

    @staticmethod
    def _global_set_key() -> str:
        return "voice:active:global"

    @staticmethod
    def _daily_quota_key(user_id: uuid.UUID, date_key: str) -> str:
        return f"voice:quota:{user_id}:{date_key}"

    def _prune_stale_global_sessions(self) -> int:
        global_key = self._global_set_key()
        members = self._redis.smembers(global_key)
        active_count = 0
        for raw_member in members:
            sid_str = (
                raw_member.decode("utf-8") if isinstance(raw_member, bytes) else str(raw_member)
            )
            try:
                sid = uuid.UUID(sid_str)
            except ValueError:
                self._redis.srem(global_key, raw_member)
                continue
            if self._redis.exists(self._session_lock_key(sid)):
                active_count += 1
            else:
                self._redis.srem(global_key, sid_str)
        return active_count

    def save_ticket(
        self,
        ticket_hash: str,
        ticket: StoredSessionTicket,
        ttl_seconds: int,
    ) -> None:
        payload = ticket.model_dump_json()
        self._redis.set(
            self._ticket_key(ticket_hash),
            payload,
            ex=max(1, ttl_seconds),
        )

    def consume_ticket(
        self,
        ticket_hash: str,
        *,
        now: datetime | None = None,
    ) -> StoredSessionTicket | None:
        key = self._ticket_key(ticket_hash)
        raw = self._redis.getdel(key)
        if not raw:
            return None
        raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        try:
            parsed = StoredSessionTicket.model_validate_json(raw_str)
        except Exception:
            return None
        effective_now = now or datetime.now(UTC)
        if parsed.expires_at <= effective_now:
            return None
        return parsed

    def is_user_session_active(self, user_id: uuid.UUID) -> bool:
        return bool(self._redis.exists(self._user_active_key(user_id)))

    def get_global_active_count(self) -> int:
        return self._prune_stale_global_sessions()

    def get_daily_session_count(
        self,
        user_id: uuid.UUID,
        *,
        date_key: str | None = None,
    ) -> int:
        d_key = date_key or current_utc_date_key()
        raw = self._redis.get(self._daily_quota_key(user_id, d_key))
        if not raw:
            return 0
        raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        try:
            return max(0, int(raw_str))
        except ValueError:
            return 0

    def acquire_session(
        self,
        session: ActiveVoiceSession,
        *,
        max_user_concurrent: int,
        max_global_concurrent: int,
        max_daily_sessions: int,
        lock_ttl_seconds: int,
        date_key: str | None = None,
    ) -> None:
        _ = max_user_concurrent
        d_key = date_key or current_utc_date_key()
        ttl = max(1, lock_ttl_seconds)

        if self.is_user_session_active(session.user_id):
            raise ConflictException(
                "You already have an active Voice Practice session.",
                code="VOICE_SESSION_ALREADY_ACTIVE",
            )

        if self._prune_stale_global_sessions() >= max_global_concurrent:
            raise ServiceUnavailableException(
                "All voice practice channels are currently in use. Please try again shortly.",
                code="VOICE_GLOBAL_CAPACITY_REACHED",
            )

        used_today = self.get_daily_session_count(session.user_id, date_key=d_key)
        if used_today >= max_daily_sessions:
            raise TooManyRequestsException(
                f"Daily voice practice limit reached ({max_daily_sessions} sessions per day).",
                code="VOICE_DAILY_QUOTA_EXCEEDED",
                details={"daily_limit": max_daily_sessions, "used_today": used_today},
            )

        user_key = self._user_active_key(session.user_id)
        acquired_user = self._redis.set(
            user_key,
            str(session.session_id),
            nx=True,
            ex=ttl,
        )
        if not acquired_user:
            raise ConflictException(
                "You already have an active Voice Practice session.",
                code="VOICE_SESSION_ALREADY_ACTIVE",
            )

        session_key = self._session_lock_key(session.session_id)
        global_key = self._global_set_key()
        quota_key = self._daily_quota_key(session.user_id, d_key)

        self._redis.set(
            session_key,
            json.dumps({"user_id": str(session.user_id), "session_id": str(session.session_id)}),
            ex=ttl,
        )
        self._redis.sadd(global_key, str(session.session_id))
        active_now = self._prune_stale_global_sessions()
        if active_now > max_global_concurrent:
            self._redis.delete(session_key)
            self._redis.delete(user_key)
            self._redis.srem(global_key, str(session.session_id))
            raise ServiceUnavailableException(
                "All voice practice channels are currently in use. Please try again shortly.",
                code="VOICE_GLOBAL_CAPACITY_REACHED",
            )

        new_quota = int(self._redis.incr(quota_key))
        if new_quota == 1:
            self._redis.expire(quota_key, 172_800)
        if new_quota > max_daily_sessions:
            self._redis.decr(quota_key)
            self._redis.delete(session_key)
            self._redis.delete(user_key)
            self._redis.srem(global_key, str(session.session_id))
            raise TooManyRequestsException(
                f"Daily voice practice limit reached ({max_daily_sessions} sessions per day).",
                code="VOICE_DAILY_QUOTA_EXCEEDED",
                details={"daily_limit": max_daily_sessions, "used_today": max_daily_sessions},
            )

    def release_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        refund_daily_quota: bool = False,
        date_key: str | None = None,
    ) -> bool:
        d_key = date_key or current_utc_date_key()
        session_key = self._session_lock_key(session_id)
        user_key = self._user_active_key(user_id)
        global_key = self._global_set_key()

        existed = bool(self._redis.delete(session_key))
        current_user_sid = self._redis.get(user_key)
        if current_user_sid:
            sid_str = (
                current_user_sid.decode("utf-8")
                if isinstance(current_user_sid, bytes)
                else str(current_user_sid)
            )
            if sid_str == str(session_id):
                self._redis.delete(user_key)
                existed = True

        removed_global = bool(self._redis.srem(global_key, str(session_id)))
        existed = existed or removed_global

        if existed and refund_daily_quota:
            quota_key = self._daily_quota_key(user_id, d_key)
            val = int(self._redis.decr(quota_key))
            if val < 0:
                self._redis.set(quota_key, 0, ex=86_400)
        return existed


class FallbackVoiceSessionStore:
    """
    Uses Redis for distributed concurrency and quota enforcement when reachable.

    PRODUCTION SAFETY INVARIANT:
    In `production` or `staging` environments (or whenever `VOICE_ALLOW_IN_MEMORY_FALLBACK` is
    False), Redis failures fail closed with `ServiceUnavailableException`
    (`VOICE_STATE_STORE_UNAVAILABLE`). Single-process in-memory fallback is permitted ONLY in
    `development` or `testing` when explicitly enabled.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        fallback_store: InMemoryVoiceSessionStore | None = None,
        redis_client: redis.Redis | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._fallback = fallback_store or _default_in_memory_store
        self._redis_client = redis_client
        self._redis_store: RedisVoiceSessionStore | None = None
        self._fallback_warned = False

    def _is_fallback_allowed(self) -> bool:
        if self.settings.APP_ENV in ("production", "staging"):
            return False
        return bool(self.settings.VOICE_ALLOW_IN_MEMORY_FALLBACK)

    def _select_store(self) -> VoiceSessionStore:
        try:
            if self._redis_client is None:
                self._redis_client = redis.Redis.from_url(
                    self.settings.REDIS_URL,
                    socket_connect_timeout=0.5,
                    socket_timeout=1.0,
                )
            self._redis_client.ping()
            if self._redis_store is None:
                self._redis_store = RedisVoiceSessionStore(self._redis_client)
            return self._redis_store
        except Exception as exc:
            if not self._is_fallback_allowed():
                logger.error(
                    "Redis is required for voice practice state and is unavailable",
                    extra={
                        "extra_data": {
                            "environment": self.settings.APP_ENV,
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                raise ServiceUnavailableException(
                    "Voice practice session coordination service is temporarily unavailable.",
                    code="VOICE_STATE_STORE_UNAVAILABLE",
                ) from exc

            if not self._fallback_warned:
                self._fallback_warned = True
                logger.warning(
                    "Redis unavailable for voice practice state; using single-process "
                    "in-memory fallback (development/testing only, not multi-worker safe)",
                    extra={"extra_data": {"error_type": type(exc).__name__}},
                )
            return self._fallback

    def save_ticket(
        self,
        ticket_hash: str,
        ticket: StoredSessionTicket,
        ttl_seconds: int,
    ) -> None:
        self._select_store().save_ticket(ticket_hash, ticket, ttl_seconds)

    def consume_ticket(
        self,
        ticket_hash: str,
        *,
        now: datetime | None = None,
    ) -> StoredSessionTicket | None:
        return self._select_store().consume_ticket(ticket_hash, now=now)

    def is_user_session_active(self, user_id: uuid.UUID) -> bool:
        return self._select_store().is_user_session_active(user_id)

    def get_global_active_count(self) -> int:
        return self._select_store().get_global_active_count()

    def get_daily_session_count(
        self,
        user_id: uuid.UUID,
        *,
        date_key: str | None = None,
    ) -> int:
        return self._select_store().get_daily_session_count(user_id, date_key=date_key)

    def acquire_session(
        self,
        session: ActiveVoiceSession,
        *,
        max_user_concurrent: int,
        max_global_concurrent: int,
        max_daily_sessions: int,
        lock_ttl_seconds: int,
        date_key: str | None = None,
    ) -> None:
        self._select_store().acquire_session(
            session,
            max_user_concurrent=max_user_concurrent,
            max_global_concurrent=max_global_concurrent,
            max_daily_sessions=max_daily_sessions,
            lock_ttl_seconds=lock_ttl_seconds,
            date_key=date_key,
        )

    def release_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        refund_daily_quota: bool = False,
        date_key: str | None = None,
    ) -> bool:
        return self._select_store().release_session(
            session_id,
            user_id,
            refund_daily_quota=refund_daily_quota,
            date_key=date_key,
        )


_default_in_memory_store = InMemoryVoiceSessionStore()
_default_fallback_store: FallbackVoiceSessionStore | None = None


def get_in_memory_voice_session_store() -> InMemoryVoiceSessionStore:
    return _default_in_memory_store


def get_voice_session_store() -> VoiceSessionStore:
    global _default_fallback_store
    if _default_fallback_store is None:
        _default_fallback_store = FallbackVoiceSessionStore()
    return _default_fallback_store


class VoiceGatewayPersistence:
    """
    Executes short-lived database transactions for WebSocket session and turn persistence
    so long-lived WebSocket connections never hold a PostgreSQL connection open.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._session_factory = session_factory or get_session_factory()
        self._lock = threading.Lock()

    def mark_active(self, session_id: uuid.UUID) -> None:
        with self._lock:
            db = self._session_factory()
            try:
                repo = VoicePracticeRepository(db)
                repo.mark_session_active(session_id)
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.warning(
                    "Failed marking voice session active in DB",
                    extra={
                        "extra_data": {
                            "session_id": str(session_id),
                            "error_type": type(exc).__name__,
                        }
                    },
                )
            finally:
                db.close()

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
    ) -> None:
        cleaned = transcript_text.strip()
        if not cleaned:
            return
        with self._lock:
            db = self._session_factory()
            try:
                repo = VoicePracticeRepository(db)
                repo.save_turn(
                    session_id=session_id,
                    turn_index=turn_index,
                    speaker_role=speaker_role,
                    transcript_text=cleaned,
                    turn_status=turn_status,
                    started_at=started_at,
                    completed_at=completed_at,
                    metadata_json=metadata_json,
                )
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error(
                    "Failed persisting voice practice turn",
                    extra={
                        "extra_data": {
                            "session_id": str(session_id),
                            "turn_index": turn_index,
                            "error_type": type(exc).__name__,
                        }
                    },
                )
            finally:
                db.close()

    def finalize_session(
        self,
        *,
        session_id: uuid.UUID,
        termination_reason: str,
        duration_seconds: float,
        had_interruption: bool = False,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock:
            db = self._session_factory()
            try:
                repo = VoicePracticeRepository(db)
                if termination_reason == "error":
                    final_status = VoiceSessionStatus.FAILED.value
                elif (
                    termination_reason == "client_disconnected"
                    or had_interruption
                    and termination_reason != "client_requested"
                ):
                    final_status = VoiceSessionStatus.INTERRUPTED.value
                else:
                    final_status = VoiceSessionStatus.COMPLETED.value

                repo.finalize_session(
                    session_id,
                    status=final_status,
                    termination_reason=termination_reason,
                    duration_seconds=duration_seconds,
                    error_code=error_code,
                    error_message=error_message,
                )
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error(
                    "Failed finalizing voice practice session in DB",
                    extra={
                        "extra_data": {
                            "session_id": str(session_id),
                            "error_type": type(exc).__name__,
                        }
                    },
                )
            finally:
                db.close()


class VoicePracticeService:
    """
    Coordinates Voice Practice connection tickets, RAG grounding, session persistence,
    history retrieval, and post-session evaluation.
    """

    def __init__(
        self,
        db: Session,
        *,
        store: VoiceSessionStore | None = None,
        retrieval_service: RetrievalService | None = None,
        evaluator: VoicePracticeEvaluator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.store = store or get_voice_session_store()
        self.auth_repo = AuthRepository(db)
        self.repo = VoicePracticeRepository(db)
        self.retrieval_service = retrieval_service or RetrievalService(
            db,
            settings=self.settings,
        )
        self.evaluator = evaluator or create_voice_practice_evaluator(self.settings)

    def _ensure_api_key_configured(self) -> None:
        if not self.settings.GEMINI_API_KEY or not self.settings.GEMINI_API_KEY.strip():
            raise ServiceUnavailableException(
                "Voice Practice AI service is not configured on the server.",
                code="VOICE_API_KEY_MISSING",
            )

    def _check_pre_ticket_limits(self, user_id: uuid.UUID) -> int:
        if self.store.is_user_session_active(user_id):
            raise ConflictException(
                "You already have an active Voice Practice session.",
                code="VOICE_SESSION_ALREADY_ACTIVE",
            )

        max_global = self.settings.VOICE_MAX_GLOBAL_CONCURRENT_SESSIONS
        if self.store.get_global_active_count() >= max_global:
            raise ServiceUnavailableException(
                "All voice practice channels are currently in use. Please try again shortly.",
                code="VOICE_GLOBAL_CAPACITY_REACHED",
            )

        used_today = self.store.get_daily_session_count(user_id)
        max_daily = self.settings.VOICE_MAX_DAILY_SESSIONS_PER_USER
        if used_today >= max_daily:
            raise TooManyRequestsException(
                f"Daily voice practice limit reached ({max_daily} sessions per day).",
                code="VOICE_DAILY_QUOTA_EXCEEDED",
                details={"daily_limit": max_daily, "used_today": used_today},
            )
        return used_today

    def _validate_and_retrieve_document_grounding(
        self,
        user_id: uuid.UUID,
        doc_config: DocumentOralPracticeConfig,
    ) -> tuple[list[Document], list[tuple[str, RetrievedChunk]], VoiceGroundingStatus]:
        doc_ids = doc_config.document_ids
        stmt = select(Document).where(
            Document.id.in_(doc_ids),
            Document.user_id == user_id,
        )
        found_docs = list(self.db.scalars(stmt).all())
        found_map = {doc.id: doc for doc in found_docs}

        missing_ids = [d_id for d_id in doc_ids if d_id not in found_map]
        if missing_ids:
            raise BadRequestException(
                "One or more selected documents do not exist or are not accessible.",
                code="DOCUMENTS_NOT_FOUND",
            )

        not_ready = [
            doc.original_filename
            for doc in found_docs
            if doc.processing_status != ProcessingStatus.READY.value
        ]
        if not_ready:
            unready_str = ", ".join(not_ready)
            raise BadRequestException(
                "Selected documents must be fully processed before oral practice. "
                f"Not ready: {unready_str}",
                code="DOCUMENTS_NOT_READY",
            )

        ordered_docs = [found_map[d_id] for d_id in doc_ids]
        query = (
            doc_config.focus_topic.strip()
            if doc_config.focus_topic and doc_config.focus_topic.strip()
            else "Core concepts, definitions, principles, and key facts from: "
            + ", ".join(d.original_filename for d in ordered_docs)
        )

        grounding_status = VoiceGroundingStatus.GROUNDED
        try:
            chunks = self.retrieval_service.retrieve_relevant_chunks(
                user_id=user_id,
                query=query,
                document_ids=doc_ids,
                limit=self.settings.VOICE_RAG_TOP_K_CHUNKS,
            )
            if not chunks:
                grounding_status = VoiceGroundingStatus.EMPTY_CONTEXT
        except Exception as exc:
            logger.warning(
                "RAG retrieval failed during document_oral setup; continuing with empty chunks",
                extra={"extra_data": {"error_type": type(exc).__name__}},
            )
            chunks = []
            grounding_status = VoiceGroundingStatus.DEGRADED_RETRIEVAL_ERROR

        ordered_chunks, _ = assign_voice_source_keys(chunks)
        return ordered_docs, ordered_chunks, grounding_status

    def issue_connection_ticket(
        self,
        user: User,
        payload: VoiceSessionTicketRequest,
    ) -> VoiceSessionTicketResponse:
        if user.status != AccountStatus.ACTIVE.value:
            raise ForbiddenException(
                "Your account is not active",
                code="ACCOUNT_INACTIVE",
            )

        self._ensure_api_key_configured()
        self.repo.reconcile_expired_pending_sessions(
            user.id,
            ttl_seconds=self.settings.VOICE_TICKET_TTL_SECONDS,
        )
        used_today = self._check_pre_ticket_limits(user.id)

        ordered_docs: list[Document] = []
        ordered_chunks: list[tuple[str, RetrievedChunk]] = []
        grounding_status = VoiceGroundingStatus.NOT_APPLICABLE
        if payload.mode == VoicePracticeMode.DOCUMENT_ORAL:
            assert payload.document_oral_config is not None
            ordered_docs, ordered_chunks, grounding_status = (
                self._validate_and_retrieve_document_grounding(
                    user.id,
                    payload.document_oral_config,
                )
            )

        system_instruction = build_live_system_instruction(
            mode=payload.mode,
            interview_config=payload.interview_config,
            communication_config=payload.communication_config,
            document_oral_config=payload.document_oral_config,
            chunks_by_key=ordered_chunks,
            max_context_chars=self.settings.VOICE_RAG_MAX_CONTEXT_CHARS,
        )

        session_id = uuid.uuid4()
        title = _build_session_title(payload, ordered_docs)
        config_dict = _build_session_config_dict(payload)
        if payload.mode == VoicePracticeMode.DOCUMENT_ORAL:
            config_dict["grounding_status"] = grounding_status.value

        try:
            self.repo.create_session(
                session_id=session_id,
                user_id=user.id,
                mode=payload.mode.value,
                title=title,
                config_json=config_dict,
                document_ids=[d.id for d in ordered_docs],
                retrieved_chunks=ordered_chunks,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        raw_ticket = generate_connection_ticket()
        ticket_hash = hash_connection_ticket(raw_ticket)
        now = datetime.now(UTC)
        ttl_seconds = self.settings.VOICE_TICKET_TTL_SECONDS
        expires_at = now + timedelta(seconds=ttl_seconds)

        stored_ticket = StoredSessionTicket(
            session_id=session_id,
            user_id=user.id,
            mode=payload.mode,
            input_sample_rate_hz=payload.input_sample_rate_hz,
            system_instruction=system_instruction,
            issued_at=now,
            expires_at=expires_at,
        )
        self.store.save_ticket(ticket_hash, stored_ticket, ttl_seconds=ttl_seconds)

        logger.info(
            "Issued Voice Practice WebSocket connection ticket",
            extra={
                "extra_data": {
                    "user_id": str(user.id),
                    "session_id": str(session_id),
                    "mode": payload.mode.value,
                    "ttl_seconds": ttl_seconds,
                }
            },
        )

        ws_path = f"{self.settings.API_V1_PREFIX}/voice-practice/ws"
        max_daily = self.settings.VOICE_MAX_DAILY_SESSIONS_PER_USER
        return VoiceSessionTicketResponse(
            ticket=raw_ticket,
            session_id=session_id,
            websocket_path=ws_path,
            expires_in_seconds=ttl_seconds,
            expires_at=expires_at,
            mode=payload.mode,
            title=title,
            grounding_status=grounding_status,
            audio_format=VoiceAudioFormatSpec(
                input_sample_rate_hz=payload.input_sample_rate_hz,
            ),
            limits=VoiceSessionLimitsSpec(
                max_duration_seconds=self.settings.VOICE_MAX_SESSION_DURATION_SECONDS,
                warning_at_seconds=self.settings.VOICE_SESSION_WARNING_SECONDS,
                max_concurrent_per_user=self.settings.VOICE_MAX_CONCURRENT_SESSIONS_PER_USER,
                daily_quota_limit=max_daily,
                daily_quota_remaining=max(0, max_daily - used_today),
                max_audio_frame_bytes=self.settings.VOICE_MAX_AUDIO_FRAME_BYTES,
            ),
        )

    def redeem_ticket_and_acquire_session(
        self,
        raw_ticket: str | None,
    ) -> ActiveVoiceSession:
        if not raw_ticket or not raw_ticket.strip():
            raise UnauthorizedException(
                "Connection ticket is required",
                code="MISSING_CONNECTION_TICKET",
            )

        cleaned = raw_ticket.strip()
        if len(cleaned) < 16 or len(cleaned) > 256:
            raise UnauthorizedException(
                "Invalid or expired connection ticket",
                code="INVALID_CONNECTION_TICKET",
            )

        ticket_hash = hash_connection_ticket(cleaned)
        stored = self.store.consume_ticket(ticket_hash)
        if stored is None:
            raise UnauthorizedException(
                "Connection ticket is invalid, expired, or has already been used",
                code="INVALID_CONNECTION_TICKET",
            )

        user = self.auth_repo.get_user_by_id(stored.user_id)
        if user is None:
            raise UnauthorizedException(
                "Authenticated user no longer exists",
                code="USER_NOT_FOUND",
            )

        if user.status != AccountStatus.ACTIVE.value:
            raise ForbiddenException(
                "Your account is not active",
                code="ACCOUNT_INACTIVE",
            )

        self._ensure_api_key_configured()

        now = datetime.now(UTC)
        active_session = ActiveVoiceSession(
            session_id=stored.session_id,
            user_id=stored.user_id,
            mode=stored.mode,
            input_sample_rate_hz=stored.input_sample_rate_hz,
            system_instruction=stored.system_instruction,
            started_at=now,
        )

        lock_ttl = self.settings.VOICE_MAX_SESSION_DURATION_SECONDS + 60
        self.store.acquire_session(
            active_session,
            max_user_concurrent=self.settings.VOICE_MAX_CONCURRENT_SESSIONS_PER_USER,
            max_global_concurrent=self.settings.VOICE_MAX_GLOBAL_CONCURRENT_SESSIONS,
            max_daily_sessions=self.settings.VOICE_MAX_DAILY_SESSIONS_PER_USER,
            lock_ttl_seconds=lock_ttl,
        )

        try:
            self.repo.mark_session_active(stored.session_id, started_at=now)
            self.db.commit()
        except Exception:
            self.db.rollback()

        logger.info(
            "Redeemed Voice Practice ticket and acquired session slot",
            extra={
                "extra_data": {
                    "user_id": str(active_session.user_id),
                    "session_id": str(active_session.session_id),
                    "mode": active_session.mode.value,
                }
            },
        )
        return active_session

    def release_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        refund_daily_quota: bool = False,
    ) -> bool:
        released = self.store.release_session(
            session_id,
            user_id,
            refund_daily_quota=refund_daily_quota,
        )
        if released:
            logger.info(
                "Released Voice Practice session resources",
                extra={
                    "extra_data": {
                        "user_id": str(user_id),
                        "session_id": str(session_id),
                        "refunded_quota": refund_daily_quota,
                    }
                },
            )
        return released

    # ========================================================================
    # SESSION HISTORY, TURNS, FEEDBACK & EVALUATION METHODS (PHASE 4)
    # ========================================================================

    def list_sessions(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
        mode: str | None = None,
        session_status: str | None = None,
    ) -> VoicePracticeSessionListResponse:
        try:
            reconciled = self.repo.reconcile_expired_pending_sessions(
                user_id,
                ttl_seconds=self.settings.VOICE_TICKET_TTL_SECONDS,
            )
            if reconciled > 0:
                self.db.commit()
        except Exception:
            self.db.rollback()

        items, total = self.repo.list_sessions_for_user(
            user_id,
            limit=limit,
            offset=offset,
            mode=mode,
            status=session_status,
        )
        return VoicePracticeSessionListResponse(
            items=[_to_session_summary_read(s) for s in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_session(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> VoicePracticeSessionDetailRead:
        try:
            reconciled = self.repo.reconcile_expired_pending_sessions(
                user_id,
                ttl_seconds=self.settings.VOICE_TICKET_TTL_SECONDS,
            )
            if reconciled > 0:
                self.db.commit()
        except Exception:
            self.db.rollback()

        session = self.repo.get_session_for_user(session_id, user_id)
        if session is None:
            raise NotFoundException(
                "Voice practice session not found",
                code="VOICE_SESSION_NOT_FOUND",
            )
        return _to_session_detail_read(session)

    def get_session_turns(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> list[VoicePracticeTurnRead]:
        session = self.repo.get_session_for_user(session_id, user_id)
        if session is None:
            raise NotFoundException(
                "Voice practice session not found",
                code="VOICE_SESSION_NOT_FOUND",
            )
        turns = self.repo.list_turns_for_session(session_id)
        return [_to_turn_read(t) for t in turns]

    def get_session_feedback(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> VoicePracticeFeedbackRead:
        session = self.repo.get_session_for_user(session_id, user_id)
        if session is None:
            raise NotFoundException(
                "Voice practice session not found",
                code="VOICE_SESSION_NOT_FOUND",
            )
        if session.feedback is None:
            raise NotFoundException(
                "Evaluation feedback has not been generated for this session yet",
                code="VOICE_FEEDBACK_NOT_FOUND",
            )
        return _to_feedback_read(session.feedback)

    def evaluate_session(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> VoicePracticeFeedbackRead:
        """
        Executes post-session evaluation using a 3-stage transaction pattern so
        no database connection or transaction is held open during the Gemini API call.
        """
        # Stage 1: Validate ownership, session state, and transcript availability
        session = self.repo.get_session_for_user(session_id, user_id)
        if session is None:
            raise NotFoundException(
                "Voice practice session not found",
                code="VOICE_SESSION_NOT_FOUND",
            )

        if session.status in (
            VoiceSessionStatus.PENDING.value,
            VoiceSessionStatus.ACTIVE.value,
        ):
            raise ConflictException(
                "Cannot evaluate a voice practice session while it is still active.",
                code="VOICE_SESSION_STILL_ACTIVE",
            )

        if session.evaluation_status == VoiceEvaluationStatus.PROCESSING.value:
            raise ConflictException(
                "Evaluation is already in progress for this session.",
                code="VOICE_EVALUATION_IN_PROGRESS",
            )

        if (
            session.evaluation_status == VoiceEvaluationStatus.COMPLETED.value
            and session.feedback is not None
            and session.feedback.evaluation_status == VoiceEvaluationStatus.COMPLETED.value
        ):
            return _to_feedback_read(session.feedback)

        turns = list(session.turns or [])
        user_turns = [t for t in turns if t.speaker_role == "user" and t.transcript_text.strip()]
        total_user_chars = sum(len(t.transcript_text.strip()) for t in user_turns)
        if not user_turns or total_user_chars < 3:
            raise BadRequestException(
                "Session does not contain enough spoken user transcript to evaluate.",
                code="INSUFFICIENT_TRANSCRIPT",
            )

        turn_inputs = [
            TurnTranscriptInput(
                turn_index=t.turn_index,
                speaker_role=t.speaker_role,
                transcript_text=t.transcript_text,
                turn_status=t.turn_status,
            )
            for t in turns
        ]

        chunks_by_key: list[tuple[str, RetrievedChunk]] = []
        for sc in session.session_chunks or []:
            if sc.chunk is not None and sc.document is not None:
                chunks_by_key.append(
                    (
                        sc.source_key,
                        RetrievedChunk(
                            chunk_id=sc.chunk.id,
                            document_id=sc.document.id,
                            original_filename=sc.document.original_filename,
                            chunk_index=sc.chunk.chunk_index,
                            page_start=sc.chunk.page_start,
                            page_end=sc.chunk.page_end,
                            page_number=sc.chunk.page_start,
                            section_title=sc.chunk.section_title,
                            text=sc.chunk.content,
                            distance=round(1.0 - sc.similarity_score, 6),
                            similarity_score=sc.similarity_score,
                        ),
                    )
                )

        mode_str = session.mode
        config_copy = dict(session.config_json or {})
        model_name = self.settings.GEMINI_GENERATION_MODEL

        try:
            claimed = self.repo.claim_session_for_evaluation(
                session_id,
                user_id,
                model_name=model_name,
            )
            if not claimed:
                self.db.rollback()
                raise ConflictException(
                    "Evaluation is already in progress for this session.",
                    code="VOICE_EVALUATION_IN_PROGRESS",
                )
            self.db.commit()
        except ConflictException:
            raise
        except Exception:
            self.db.rollback()
            raise

        # Close transaction/session state before external LLM network I/O
        self.db.close()

        # Stage 2: External Gemini Evaluation Call (No DB transaction open)
        try:
            evaluated = self.evaluator.evaluate_session(
                mode=mode_str,
                configuration=config_copy,
                turns=turn_inputs,
                chunks=chunks_by_key,
            )
        except VoiceEvaluationError as exc:
            try:
                self.repo.save_feedback_failed(
                    session_id,
                    model_name=model_name,
                    error_code=exc.code,
                    error_message=exc.user_message,
                )
                self.db.commit()
            except Exception:
                self.db.rollback()

            if exc.code == "INSUFFICIENT_TRANSCRIPT":
                raise BadRequestException(exc.user_message, code=exc.code) from exc
            raise ServiceUnavailableException(exc.user_message, code=exc.code) from exc
        except Exception as exc:
            try:
                self.repo.save_feedback_failed(
                    session_id,
                    model_name=model_name,
                    error_code="VOICE_EVALUATION_FAILED",
                    error_message="Unexpected error during voice practice evaluation.",
                )
                self.db.commit()
            except Exception:
                self.db.rollback()
            raise ServiceUnavailableException(
                "Unable to complete voice practice evaluation. Please try again later.",
                code="VOICE_EVALUATION_FAILED",
            ) from exc

        # Stage 3: Short Atomic Write Transaction
        structured_payload: dict[str, Any] = {
            "summary": evaluated.summary,
            "has_sufficient_transcript": evaluated.has_sufficient_transcript,
            "transcript_evidence": [
                e.model_dump(mode="json") for e in evaluated.transcript_evidence
            ],
            "missing_concepts": evaluated.missing_concepts,
            "unsupported_statements": evaluated.unsupported_statements,
            "source_references": [r.model_dump(mode="json") for r in evaluated.source_references],
        }
        dims_dump = [d.model_dump(mode="json") for d in evaluated.dimension_scores]

        try:
            saved_feedback = self.repo.save_feedback_completed(
                session_id,
                model_name=evaluated.model_name,
                overall_score=evaluated.overall_score,
                dimension_scores=dims_dump,
                strengths=evaluated.strengths,
                areas_for_improvement=evaluated.areas_for_improvement,
                recommendations=evaluated.recommendations,
                structured_feedback=structured_payload,
            )
            self.db.commit()
            self.db.refresh(saved_feedback)
        except Exception:
            self.db.rollback()
            raise

        return _to_feedback_read(saved_feedback)

    def delete_session(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
    ) -> VoicePracticeSessionDeleteResponse:
        session = self.repo.get_session_for_user(session_id, user_id)
        if session is None:
            raise NotFoundException(
                "Voice practice session not found",
                code="VOICE_SESSION_NOT_FOUND",
            )
        if (
            self.store.is_user_session_active(user_id)
            and session.status == VoiceSessionStatus.ACTIVE.value
        ):
            raise ConflictException(
                "Cannot delete an active voice practice session.",
                code="VOICE_SESSION_STILL_ACTIVE",
            )
        try:
            self.repo.delete_session(session)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return VoicePracticeSessionDeleteResponse(
            deleted=True,
            session_id=session_id,
        )
