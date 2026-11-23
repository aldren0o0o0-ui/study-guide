import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from tests.integration.test_voice_practice_gateway import (
    FakeGeminiLiveSession,
    _make_server_message,
    _register_and_get_headers,
)

from app.core.config import Settings, get_settings
from app.main import create_app
from app.modules.auth.models import User
from app.modules.documents.models import Document, DocumentChunk, ProcessingStatus
from app.modules.processing.retrieval import RetrievalService, RetrievedChunk
from app.modules.voice_practice.evaluator import (
    EvaluatedVoiceSession,
    TurnTranscriptInput,
    VoiceEvaluationError,
    VoicePracticeEvaluator,
    resolve_evaluation_source_references,
)
from app.modules.voice_practice.gateway import (
    GeminiLiveClientFactory,
    GeminiLiveSessionProtocol,
    get_gemini_live_client_factory,
)
from app.modules.voice_practice.models import (
    VoiceEvaluationStatus,
    VoicePracticeFeedback,
    VoicePracticeSessionChunk,
    VoicePracticeSessionDocument,
    VoicePracticeTurn,
    VoiceSessionStatus,
)
from app.modules.voice_practice.repository import VoicePracticeRepository
from app.modules.voice_practice.schemas import (
    ActiveVoiceSession,
    VoiceDimensionScore,
    VoiceTranscriptEvidence,
)
from app.modules.voice_practice.service import (
    InMemoryVoiceSessionStore,
    get_voice_session_store,
)
from app.modules.voice_practice.ws_router import get_voice_practice_evaluator


class FakeVoicePracticeEvaluator:
    """Controllable evaluator double for deterministic Phase 4 REST & DB integration tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raise_error: Exception | None = None
        self.overall_score: float = 86.5

    def evaluate_session(
        self,
        *,
        mode: str,
        configuration: dict[str, Any],
        turns: Sequence[TurnTranscriptInput],
        chunks: Sequence[tuple[str, RetrievedChunk]] | None = None,
    ) -> EvaluatedVoiceSession:
        self.calls.append(
            {
                "mode": mode,
                "configuration": configuration,
                "turns": list(turns),
                "chunks": list(chunks or []),
            }
        )
        if self.raise_error is not None:
            raise self.raise_error

        key_map = dict(chunks or [])
        cited_keys = list(key_map.keys())[:2]
        refs = resolve_evaluation_source_references(cited_keys, key_map=key_map)

        if mode == "interview":
            dims = [
                VoiceDimensionScore(
                    dimension="Relevance",
                    score=90.0,
                    rationale="Directly addressed the technical question.",
                ),
                VoiceDimensionScore(
                    dimension="Technical or Factual Correctness",
                    score=85.0,
                    rationale="Accurate explanation of indexing and transactions.",
                ),
            ]
        elif mode == "communication":
            dims = [
                VoiceDimensionScore(
                    dimension="Clarity",
                    score=88.0,
                    rationale="Spoke with clear organization and transitions.",
                ),
                VoiceDimensionScore(
                    dimension="Logical Flow",
                    score=84.0,
                    rationale="Well-sequenced main points.",
                ),
            ]
        else:
            dims = [
                VoiceDimensionScore(
                    dimension="Factual Accuracy",
                    score=89.0,
                    rationale="Grounded in the selected source chunks.",
                ),
                VoiceDimensionScore(
                    dimension="Concept Coverage",
                    score=84.0,
                    rationale="Covered primary definitions from S1.",
                ),
            ]

        first_user_turn = next((t for t in turns if t.speaker_role == "user"), turns[0])
        return EvaluatedVoiceSession(
            overall_score=self.overall_score,
            summary=f"Solid {mode} performance with clear spoken articulation.",
            dimension_scores=dims,
            strengths=["Clear structure", "Accurate core terminology"],
            areas_for_improvement=["Include concrete metrics or edge cases"],
            recommendations=["Practice summarizing key takeaways in 30 seconds"],
            transcript_evidence=[
                VoiceTranscriptEvidence(
                    turn_index=first_user_turn.turn_index,
                    quote=first_user_turn.transcript_text[:120],
                    observation="Demonstrated understanding of the prompt.",
                )
            ],
            missing_concepts=["Write-Ahead Logging (WAL)"] if mode == "document_oral" else [],
            unsupported_statements=[],
            source_references=refs,
            model_name="gemini-2.5-flash",
            has_sufficient_transcript=True,
        )


def _seed_user_document_with_chunks(
    db: Session,
    *,
    user_id: uuid.UUID,
    filename: str = "database_internals.pdf",
    processing_status: str = ProcessingStatus.READY.value,
    chunk_texts: list[str] | None = None,
) -> tuple[Document, list[DocumentChunk]]:
    import hashlib

    now = datetime.now(UTC)
    texts = chunk_texts or [
        "PostgreSQL MVCC uses transaction IDs (xmin, xmax) to provide snapshot isolation.",
        "B-tree indexes support equality and range queries efficiently on ordered keys.",
    ]
    doc_uid = uuid.uuid4()
    doc = Document(
        user_id=user_id,
        original_filename=filename,
        storage_key=f"uploads/{user_id}/{doc_uid}.pdf",
        file_extension="pdf",
        mime_type="application/pdf",
        file_size_bytes=20480,
        checksum_sha256=hashlib.sha256(f"{doc_uid}:{filename}".encode()).hexdigest(),
        processing_status=processing_status,
        chunk_count=len(texts),
        processed_at=now if processing_status == ProcessingStatus.READY.value else None,
    )
    db.add(doc)
    db.flush()

    chunks: list[DocumentChunk] = []
    # Use a deterministic normalized embedding vector (768 dims)
    vec = [1.0] + [0.0] * 767
    for idx, content in enumerate(texts):
        chunk = DocumentChunk(
            document_id=doc.id,
            user_id=user_id,
            chunk_index=idx,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            char_count=len(content),
            page_start=idx + 1,
            page_end=idx + 1,
            section_title=f"Section {idx + 1}",
            embedding=vec,
        )
        db.add(chunk)
        chunks.append(chunk)

    db.commit()
    for c in chunks:
        db.refresh(c)
    db.refresh(doc)
    return doc, chunks


@pytest.fixture
def phase4_store() -> InMemoryVoiceSessionStore:
    return InMemoryVoiceSessionStore()


@pytest.fixture
def phase4_settings() -> Settings:
    return Settings(
        APP_ENV="testing",
        GEMINI_API_KEY="mock-gemini-api-key-phase4-test",
        GEMINI_LIVE_MODEL="gemini-2.5-flash-native-audio-preview-12-2025",
        GEMINI_GENERATION_MODEL="gemini-2.5-flash",
        VOICE_TICKET_TTL_SECONDS=60,
        VOICE_MAX_SESSION_DURATION_SECONDS=600,
        VOICE_SESSION_WARNING_SECONDS=570,
        VOICE_MAX_CONCURRENT_SESSIONS_PER_USER=1,
        VOICE_MAX_GLOBAL_CONCURRENT_SESSIONS=3,
        VOICE_MAX_DAILY_SESSIONS_PER_USER=10,
        VOICE_MAX_AUDIO_FRAME_BYTES=8192,
        VOICE_MAX_JSON_MESSAGE_BYTES=4096,
    )


@pytest.fixture
def phase4_evaluator() -> FakeVoicePracticeEvaluator:
    return FakeVoicePracticeEvaluator()


@pytest.fixture
def phase4_app_factory(
    db_session: Session,
    phase4_store: InMemoryVoiceSessionStore,
    phase4_settings: Settings,
    phase4_evaluator: FakeVoicePracticeEvaluator,
) -> Any:
    _ = db_session

    def _build(
        *,
        live_factory: GeminiLiveClientFactory | None = None,
        custom_settings: Settings | None = None,
        custom_store: InMemoryVoiceSessionStore | None = None,
        custom_evaluator: VoicePracticeEvaluator | None = None,
    ) -> Any:
        active_settings = custom_settings or phase4_settings
        active_store = custom_store or phase4_store
        active_evaluator = custom_evaluator or phase4_evaluator

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: active_settings
        app.dependency_overrides[get_voice_session_store] = lambda: active_store
        app.dependency_overrides[get_voice_practice_evaluator] = lambda: active_evaluator

        if live_factory is not None:
            app.dependency_overrides[get_gemini_live_client_factory] = lambda: live_factory
        else:

            @asynccontextmanager
            async def _default_fake_factory(
                _session: ActiveVoiceSession,
                _cfg: Settings,
            ) -> AsyncIterator[GeminiLiveSessionProtocol]:
                fake = FakeGeminiLiveSession()
                try:
                    yield fake
                finally:
                    await fake.close()

            app.dependency_overrides[get_gemini_live_client_factory] = lambda: _default_fake_factory
        return app

    return _build


# ============================================================================
# 1. PRE-IMPLEMENTATION REVIEW FIXES (SAMPLE RATE NORMALIZATION & DISCONNECT)
# ============================================================================


def test_gateway_normalizes_48khz_and_24khz_pcm_to_16khz_and_rejects_unsupported(
    phase4_app_factory: Any,
) -> None:
    fake_session = FakeGeminiLiveSession(turns=[], hold_open_after_turns=True)

    @asynccontextmanager
    async def _factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        try:
            yield fake_session
        finally:
            await fake_session.close()

    app = phase4_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "resample.user@example.com")
        res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
            json={"mode": "communication", "input_sample_rate_hz": 48000},
        )
        assert res.status_code == 201
        ticket = res.json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            ready = ws.receive_json()
            assert ready["event"] == "session_ready"
            assert ready["payload"]["audio_format"]["input_sample_rate_hz"] == 48000

            # Send 480 samples at 48kHz (960 bytes) -> normalized to 160 samples at 16kHz
            ws.send_bytes(b"\x10\x01" * 480)
            started = ws.receive_json()
            assert started["event"] == "session_started"

            # Unsupported sample rate in audio_input_started is rejected
            ws.send_json({"event": "audio_input_started", "payload": {"sample_rate_hz": 8000}})
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "UNSUPPORTED_SAMPLE_RATE"
            assert err["payload"]["fatal"] is False

            # Switch to 24kHz and send 240 samples -> normalized to 160 samples at 16kHz
            ws.send_json({"event": "audio_input_started", "payload": {"sample_rate_hz": 24000}})
            in_started = ws.receive_json()
            assert in_started["event"] == "audio_input_started"
            ws.send_bytes(b"\x20\x02" * 240)

            ws.send_json({"event": "session_end"})
            ended = ws.receive_json()
            assert ended["event"] == "session_ended"

        assert len(fake_session.sent_audio_frames) == 2
        frame_1_bytes, frame_1_rate = fake_session.sent_audio_frames[0]
        frame_2_bytes, frame_2_rate = fake_session.sent_audio_frames[1]
        assert frame_1_rate == 16000
        assert len(frame_1_bytes) == 320
        assert frame_2_rate == 16000
        assert len(frame_2_bytes) == 320


def test_unexpected_client_disconnect_persists_completed_and_in_progress_turns(
    phase4_app_factory: Any,
    db_session: Session,
) -> None:
    # Turn 1 completes; Turn 2 has partial transcript when client disconnects abruptly
    turn_1 = [
        _make_server_message(
            user_transcript="Tell me about distributed locks.",
            user_finished=True,
            assistant_transcript="Distributed locks coordinate mutual exclusion across nodes.",
            assistant_finished=True,
            turn_complete=True,
        ),
    ]
    turn_2_partial = [
        _make_server_message(
            user_transcript="What happens if a lock holder crashes",
            user_finished=False,
        ),
    ]

    fake_session = FakeGeminiLiveSession(
        turns=[turn_1, turn_2_partial],
        hold_open_after_turns=True,
    )

    @asynccontextmanager
    async def _factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        try:
            yield fake_session
        finally:
            await fake_session.close()

    app = phase4_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "disconnect.persist@example.com")
        ticket_data = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
            json={"mode": "interview"},
        ).json()["data"]
        ticket = ticket_data["ticket"]
        session_id = uuid.UUID(ticket_data["session_id"])

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            _ = ws.receive_json()  # session_ready
            seen_partial_turn_2 = False
            while not seen_partial_turn_2:
                evt = ws.receive_json()
                if evt["event"] == "transcript_partial" and evt["payload"]["turn_index"] == 2:
                    seen_partial_turn_2 = True
            # Abruptly close WebSocket without sending `session_end`

        # Verify session status is `interrupted` and all 3 turns were persisted
        detail_res = client.get(
            f"/api/v1/voice-practice/sessions/{session_id}",
            headers=headers,
        )
        assert detail_res.status_code == 200
        detail = detail_res.json()["data"]
        assert detail["status"] == VoiceSessionStatus.INTERRUPTED.value
        assert detail["termination_reason"] == "client_disconnected"
        assert detail["evaluation_status"] == VoiceEvaluationStatus.PENDING.value

        turns = detail["turns"]
        assert len(turns) == 3
        assert turns[0]["turn_index"] == 1
        assert turns[0]["speaker_role"] == "user"
        assert turns[0]["transcript_text"] == "Tell me about distributed locks."
        assert turns[0]["turn_status"] == "completed"

        assert turns[1]["turn_index"] == 2
        assert turns[1]["speaker_role"] == "assistant"
        assert (
            turns[1]["transcript_text"]
            == "Distributed locks coordinate mutual exclusion across nodes."
        )
        assert turns[1]["turn_status"] == "completed"

        assert turns[2]["turn_index"] == 3
        assert turns[2]["speaker_role"] == "user"
        assert turns[2]["transcript_text"] == "What happens if a lock holder crashes"
        assert turns[2]["turn_status"] == "interrupted"
    _ = db_session


# ============================================================================
# 2. DATABASE & REPOSITORY PERSISTENCE TESTS (IDEMPOTENCY & CASCADE DELETE)
# ============================================================================


def test_repository_idempotent_save_turn_and_cascade_deletion(
    db_session: Session,
) -> None:
    user = User(
        email="repo.test@example.com",
        password_hash="hashed",
        status="active",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    doc, chunks = _seed_user_document_with_chunks(db_session, user_id=user.id)
    retrieved_chunk = RetrievedChunk(
        chunk_id=chunks[0].id,
        document_id=doc.id,
        original_filename=doc.original_filename,
        chunk_index=0,
        page_start=1,
        page_end=1,
        page_number=1,
        section_title="Section 1",
        text=chunks[0].content,
        distance=0.1,
        similarity_score=0.9,
    )

    repo = VoicePracticeRepository(db_session)
    session_id = uuid.uuid4()
    repo.create_session(
        session_id=session_id,
        user_id=user.id,
        mode="document_oral",
        title="DB Cascade Test",
        config_json={"mode": "document_oral"},
        document_ids=[doc.id],
        retrieved_chunks=[("S1", retrieved_chunk)],
    )
    db_session.commit()

    # Save turn_index=1 twice -> must update in-place without duplicate row or IntegrityError
    repo.save_turn(
        session_id=session_id,
        turn_index=1,
        speaker_role="user",
        transcript_text="Initial transcript",
        turn_status="completed",
    )
    db_session.commit()

    repo.save_turn(
        session_id=session_id,
        turn_index=1,
        speaker_role="user",
        transcript_text="Updated transcript",
        turn_status="interrupted",
    )
    db_session.commit()

    persisted_turns = repo.list_turns_for_session(session_id)
    assert len(persisted_turns) == 1
    assert persisted_turns[0].transcript_text == "Updated transcript"
    assert persisted_turns[0].turn_status == "interrupted"

    repo.save_feedback_completed(
        session_id,
        model_name="gemini-2.5-flash",
        overall_score=90.0,
        dimension_scores=[{"dimension": "Accuracy", "score": 90.0, "rationale": "Good"}],
        strengths=["Accurate"],
        areas_for_improvement=[],
        recommendations=[],
        structured_feedback={"summary": "Great job"},
    )
    db_session.commit()

    # Verify child records exist before delete
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeTurn)
            .where(VoicePracticeTurn.session_id == session_id)
        )
        == 1
    )
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeFeedback)
            .where(VoicePracticeFeedback.session_id == session_id)
        )
        == 1
    )
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeSessionDocument)
            .where(VoicePracticeSessionDocument.session_id == session_id)
        )
        == 1
    )
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeSessionChunk)
            .where(VoicePracticeSessionChunk.session_id == session_id)
        )
        == 1
    )

    # Delete session and verify cascade cleanup of child records
    session_obj = repo.get_session_for_user(session_id, user.id)
    assert session_obj is not None
    repo.delete_session(session_obj)
    db_session.commit()

    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeTurn)
            .where(VoicePracticeTurn.session_id == session_id)
        )
        == 0
    )
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeFeedback)
            .where(VoicePracticeFeedback.session_id == session_id)
        )
        == 0
    )
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeSessionDocument)
            .where(VoicePracticeSessionDocument.session_id == session_id)
        )
        == 0
    )
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(VoicePracticeSessionChunk)
            .where(VoicePracticeSessionChunk.session_id == session_id)
        )
        == 0
    )
    # Underlying user document and chunks remain intact
    assert db_session.get(Document, doc.id) is not None


# ============================================================================
# 3. THREE PRACTICE MODES & RAG DOCUMENT GROUNDING TESTS
# ============================================================================


def test_practice_modes_validation_and_rag_document_grounding(
    phase4_app_factory: Any,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_sessions: list[ActiveVoiceSession] = []

    @asynccontextmanager
    async def _capturing_factory(
        session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        captured_sessions.append(session)
        fake = FakeGeminiLiveSession(turns=[], hold_open_after_turns=True)
        try:
            yield fake
        finally:
            await fake.close()

    app = phase4_app_factory(live_factory=_capturing_factory)
    with TestClient(app) as client:
        headers_u1, _, u1_id = _register_and_get_headers(client, "rag.owner@example.com")
        headers_u2, _, u2_id = _register_and_get_headers(client, "rag.other@example.com")

        ready_doc, ready_chunks = _seed_user_document_with_chunks(
            db_session,
            user_id=u1_id,
            filename="operating_systems.pdf",
            processing_status=ProcessingStatus.READY.value,
        )
        unready_doc, _ = _seed_user_document_with_chunks(
            db_session,
            user_id=u1_id,
            filename="unprocessed_notes.pdf",
            processing_status=ProcessingStatus.PROCESSING.value,
        )
        other_user_doc, _ = _seed_user_document_with_chunks(
            db_session,
            user_id=u2_id,
            filename="private_other_user.pdf",
            processing_status=ProcessingStatus.READY.value,
        )

        # Mock RetrievalService.retrieve_relevant_chunks to return deterministic chunks
        def _fake_retrieve(
            _self: RetrievalService,
            *,
            user_id: uuid.UUID,
            query: str,
            document_ids: Sequence[uuid.UUID] | None = None,
            limit: int | None = None,
        ) -> list[RetrievedChunk]:
            _ = (user_id, query, document_ids, limit)
            return [
                RetrievedChunk(
                    chunk_id=ready_chunks[0].id,
                    document_id=ready_doc.id,
                    original_filename=ready_doc.original_filename,
                    chunk_index=0,
                    page_start=1,
                    page_end=1,
                    page_number=1,
                    section_title="Section 1",
                    text=ready_chunks[0].content,
                    distance=0.08,
                    similarity_score=0.92,
                )
            ]

        monkeypatch.setattr(RetrievalService, "retrieve_relevant_chunks", _fake_retrieve)

        # 1. Mode mismatch validation (passing interview_config in communication mode -> 422)
        mismatch_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers_u1,
            json={
                "mode": "communication",
                "interview_config": {"target_role": "Backend Engineer"},
            },
        )
        assert mismatch_res.status_code == 422

        # 2. document_oral mode without document_ids -> 422
        missing_docs_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers_u1,
            json={
                "mode": "document_oral",
                "document_oral_config": {"document_ids": []},
            },
        )
        assert missing_docs_res.status_code == 422

        # 3. Cross-user document isolation (u1 cannot select u2's document -> 400)
        cross_user_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers_u1,
            json={
                "mode": "document_oral",
                "document_oral_config": {"document_ids": [str(other_user_doc.id)]},
            },
        )
        assert cross_user_res.status_code == 400
        assert cross_user_res.json()["error"]["code"] == "DOCUMENTS_NOT_FOUND"

        # 4. Unprocessed document rejection -> 400 DOCUMENTS_NOT_READY
        unready_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers_u1,
            json={
                "mode": "document_oral",
                "document_oral_config": {"document_ids": [str(unready_doc.id)]},
            },
        )
        assert unready_res.status_code == 400
        assert unready_res.json()["error"]["code"] == "DOCUMENTS_NOT_READY"

        # 5. Valid document_oral session creation & WebSocket connection
        valid_doc_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers_u1,
            json={
                "mode": "document_oral",
                "document_oral_config": {
                    "document_ids": [str(ready_doc.id)],
                    "focus_topic": "MVCC Snapshot Isolation",
                    "difficulty": "hard",
                    "question_count": 4,
                },
            },
        )
        assert valid_doc_res.status_code == 201
        doc_ticket_data = valid_doc_res.json()["data"]
        doc_session_id = uuid.UUID(doc_ticket_data["session_id"])
        assert doc_ticket_data["mode"] == "document_oral"
        assert "MVCC Snapshot Isolation" in doc_ticket_data["title"]

        with client.websocket_connect(
            f"/api/v1/voice-practice/ws?ticket={doc_ticket_data['ticket']}"
        ) as ws:
            ready_evt = ws.receive_json()
            assert ready_evt["event"] == "session_ready"
            assert ready_evt["payload"]["mode"] == "document_oral"
            ws.send_json({"event": "session_end"})
            _ = ws.receive_json()

        assert len(captured_sessions) == 1
        assert captured_sessions[0].system_instruction is not None
        assert "operating_systems.pdf" in captured_sessions[0].system_instruction
        assert "PostgreSQL MVCC" in captured_sessions[0].system_instruction

        # Verify source document & chunk references persisted on session
        session_detail = client.get(
            f"/api/v1/voice-practice/sessions/{doc_session_id}",
            headers=headers_u1,
        ).json()["data"]
        assert len(session_detail["source_documents"]) == 1
        assert session_detail["source_documents"][0]["document_id"] == str(ready_doc.id)
        assert session_detail["source_documents"][0]["original_filename"] == "operating_systems.pdf"

        # 6. Retrieval failure handling (graceful fallback to empty chunks)
        def _failing_retrieve(*_args: Any, **_kwargs: Any) -> list[RetrievedChunk]:
            raise RuntimeError("Simulated embedding provider outage")

        monkeypatch.setattr(RetrievalService, "retrieve_relevant_chunks", _failing_retrieve)
        fallback_doc_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers_u1,
            json={
                "mode": "document_oral",
                "document_oral_config": {"document_ids": [str(ready_doc.id)]},
            },
        )
        assert fallback_doc_res.status_code == 201


# ============================================================================
# 4. POST-SESSION EVALUATION, SAFE RETRY & REST ENDPOINTS TESTS
# ============================================================================


def test_full_lifecycle_evaluation_retry_and_rest_endpoints_across_modes(
    phase4_app_factory: Any,
    phase4_evaluator: FakeVoicePracticeEvaluator,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn_messages = [
        [
            _make_server_message(
                user_transcript="Can we begin the practice session?",
                user_finished=True,
                assistant_transcript="Yes! Explain how PostgreSQL MVCC works.",
                assistant_finished=True,
                turn_complete=True,
            )
        ],
        [
            _make_server_message(
                user_transcript=(
                    "PostgreSQL MVCC uses xmin and xmax transaction IDs for snapshot isolation."
                ),
                user_finished=True,
                assistant_transcript="Great explanation of tuple visibility.",
                assistant_finished=True,
                turn_complete=True,
            )
        ],
    ]

    @asynccontextmanager
    async def _factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        fake = FakeGeminiLiveSession(turns=list(turn_messages), hold_open_after_turns=True)
        try:
            yield fake
        finally:
            await fake.close()

    app = phase4_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers_u1, _, u1_id = _register_and_get_headers(client, "eval.owner@example.com")
        headers_u2, _, _ = _register_and_get_headers(client, "eval.intruder@example.com")

        doc, chunks = _seed_user_document_with_chunks(
            db_session,
            user_id=u1_id,
            filename="postgres_guide.pdf",
        )

        monkeypatch.setattr(
            RetrievalService,
            "retrieve_relevant_chunks",
            lambda _self, **_kw: [
                RetrievedChunk(
                    chunk_id=chunks[0].id,
                    document_id=doc.id,
                    original_filename=doc.original_filename,
                    chunk_index=0,
                    page_start=1,
                    page_end=1,
                    page_number=1,
                    section_title="Section 1",
                    text=chunks[0].content,
                    distance=0.05,
                    similarity_score=0.95,
                )
            ],
        )

        # Create and run a `document_oral` session
        t_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers_u1,
            json={
                "mode": "document_oral",
                "document_oral_config": {
                    "document_ids": [str(doc.id)],
                    "focus_topic": "MVCC",
                },
            },
        )
        assert t_res.status_code == 201
        session_id = t_res.json()["data"]["session_id"]
        ticket = t_res.json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            _ = ws.receive_json()  # session_ready

            # Cannot evaluate while session is active -> 409 VOICE_SESSION_STILL_ACTIVE
            active_eval_res = client.post(
                f"/api/v1/voice-practice/sessions/{session_id}/evaluate",
                headers=headers_u1,
            )
            assert active_eval_res.status_code == 409
            assert active_eval_res.json()["error"]["code"] == "VOICE_SESSION_STILL_ACTIVE"

            # Cannot delete while session is active -> 409 VOICE_SESSION_STILL_ACTIVE
            active_del_res = client.delete(
                f"/api/v1/voice-practice/sessions/{session_id}",
                headers=headers_u1,
            )
            assert active_del_res.status_code == 409

            completed_turns = 0
            while completed_turns < 2:
                evt = ws.receive_json()
                if evt["event"] == "turn_completed":
                    completed_turns += 1

            ws.send_json({"event": "session_end"})
            ended = ws.receive_json()
            assert ended["event"] == "session_ended"

        # 1. GET /turns returns all 4 finalized turns in exact order
        turns_res = client.get(
            f"/api/v1/voice-practice/sessions/{session_id}/turns",
            headers=headers_u1,
        )
        assert turns_res.status_code == 200
        turns_data = turns_res.json()["data"]
        assert len(turns_data) == 4
        assert [t["turn_index"] for t in turns_data] == [1, 2, 3, 4]
        assert [t["speaker_role"] for t in turns_data] == ["user", "assistant", "user", "assistant"]

        # 2. GET /feedback before evaluation returns 404 VOICE_FEEDBACK_NOT_FOUND
        pre_fb_res = client.get(
            f"/api/v1/voice-practice/sessions/{session_id}/feedback",
            headers=headers_u1,
        )
        assert pre_fb_res.status_code == 404
        assert pre_fb_res.json()["error"]["code"] == "VOICE_FEEDBACK_NOT_FOUND"

        # 3. Simulate Gemini evaluation failure -> 503, marks feedback `failed`
        phase4_evaluator.raise_error = VoiceEvaluationError(
            "Provider timeout",
            code="VOICE_EVALUATION_PROVIDER_UNAVAILABLE",
            user_message="AI evaluation service is temporarily unavailable.",
            retryable=True,
        )
        fail_eval_res = client.post(
            f"/api/v1/voice-practice/sessions/{session_id}/evaluate",
            headers=headers_u1,
        )
        assert fail_eval_res.status_code == 503
        assert fail_eval_res.json()["error"]["code"] == "VOICE_EVALUATION_PROVIDER_UNAVAILABLE"

        failed_fb_res = client.get(
            f"/api/v1/voice-practice/sessions/{session_id}/feedback",
            headers=headers_u1,
        )
        assert failed_fb_res.status_code == 200
        failed_fb = failed_fb_res.json()["data"]
        assert failed_fb["evaluation_status"] == VoiceEvaluationStatus.FAILED.value
        assert failed_fb["overall_score"] is None
        assert failed_fb["error_code"] == "VOICE_EVALUATION_PROVIDER_UNAVAILABLE"

        # 4. Safe evaluation retry succeeds and updates feedback in-place without duplicate rows
        phase4_evaluator.raise_error = None
        retry_eval_res = client.post(
            f"/api/v1/voice-practice/sessions/{session_id}/evaluate",
            headers=headers_u1,
        )
        assert retry_eval_res.status_code == 200
        completed_fb = retry_eval_res.json()["data"]
        assert completed_fb["evaluation_status"] == VoiceEvaluationStatus.COMPLETED.value
        assert completed_fb["overall_score"] == 86.5
        assert len(completed_fb["dimension_scores"]) == 2
        assert len(completed_fb["source_references"]) == 1
        assert completed_fb["source_references"][0]["document_id"] == str(doc.id)
        assert completed_fb["source_references"][0]["chunk_id"] == str(chunks[0].id)
        assert completed_fb["error_code"] is None

        # Calling POST /evaluate again when completed is idempotent and returns cached feedback
        cached_eval_res = client.post(
            f"/api/v1/voice-practice/sessions/{session_id}/evaluate",
            headers=headers_u1,
        )
        assert cached_eval_res.status_code == 200
        assert len(phase4_evaluator.calls) == 2  # Not called a 3rd time

        # 5. GET /sessions list endpoint with filtering
        list_res = client.get(
            "/api/v1/voice-practice/sessions?mode=document_oral&status=completed",
            headers=headers_u1,
        )
        assert list_res.status_code == 200
        list_data = list_res.json()["data"]
        assert list_data["total"] == 1
        assert list_data["items"][0]["id"] == session_id
        assert list_data["items"][0]["turns_count"] == 4
        assert list_data["items"][0]["documents_count"] == 1
        assert list_data["items"][0]["overall_score"] == 86.5

        # 6. Cross-user access rejection across all endpoints -> 404 VOICE_SESSION_NOT_FOUND
        assert (
            client.get(
                f"/api/v1/voice-practice/sessions/{session_id}",
                headers=headers_u2,
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/api/v1/voice-practice/sessions/{session_id}/turns",
                headers=headers_u2,
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/api/v1/voice-practice/sessions/{session_id}/feedback",
                headers=headers_u2,
            ).status_code
            == 404
        )
        assert (
            client.post(
                f"/api/v1/voice-practice/sessions/{session_id}/evaluate",
                headers=headers_u2,
            ).status_code
            == 404
        )
        assert (
            client.delete(
                f"/api/v1/voice-practice/sessions/{session_id}",
                headers=headers_u2,
            ).status_code
            == 404
        )

        # 7. DELETE /sessions/{session_id} by owner succeeds and removes session
        del_res = client.delete(
            f"/api/v1/voice-practice/sessions/{session_id}",
            headers=headers_u1,
        )
        assert del_res.status_code == 200
        assert del_res.json()["data"]["deleted"] is True

        assert (
            client.get(
                f"/api/v1/voice-practice/sessions/{session_id}",
                headers=headers_u1,
            ).status_code
            == 404
        )


def test_evaluate_session_with_no_user_transcript_rejected(
    phase4_app_factory: Any,
) -> None:
    app = phase4_app_factory()
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "empty.transcript@example.com")
        t_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
            json={
                "mode": "interview",
                "interview_config": {
                    "interview_type": "technical",
                    "target_role": "Backend Engineer",
                    "experience_level": "mid",
                    "difficulty": "medium",
                },
            },
        )
        assert t_res.status_code == 201
        session_id = t_res.json()["data"]["session_id"]
        ticket = t_res.json()["data"]["ticket"]

        # Connect and immediately end session with 0 spoken turns
        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            _ = ws.receive_json()
            ws.send_json({"event": "session_end"})
            _ = ws.receive_json()

        # Evaluation is rejected with 400 INSUFFICIENT_TRANSCRIPT (never fabricates a score)
        eval_res = client.post(
            f"/api/v1/voice-practice/sessions/{session_id}/evaluate",
            headers=headers,
        )
        assert eval_res.status_code == 400
        assert eval_res.json()["error"]["code"] == "INSUFFICIENT_TRANSCRIPT"


def test_interview_and_communication_mode_evaluations(
    phase4_app_factory: Any,
) -> None:
    single_turn = [
        [
            _make_server_message(
                user_transcript=(
                    "In my previous project, I reduced API latency by 40 percent "
                    "using Redis caching."
                ),
                user_finished=True,
                assistant_transcript="Could you elaborate on your cache invalidation strategy?",
                assistant_finished=True,
                turn_complete=True,
            )
        ]
    ]

    @asynccontextmanager
    async def _factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        fake = FakeGeminiLiveSession(turns=list(single_turn), hold_open_after_turns=True)
        try:
            yield fake
        finally:
            await fake.close()

    app = phase4_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "modes.eval@example.com")

        for mode_payload, expected_dim in [
            (
                {
                    "mode": "interview",
                    "interview_config": {
                        "interview_type": "behavioral",
                        "target_role": "Backend Engineer",
                        "experience_level": "mid",
                        "difficulty": "medium",
                    },
                },
                "Relevance",
            ),
            (
                {
                    "mode": "communication",
                    "communication_config": {
                        "scenario": "impromptu_speaking",
                        "speaking_topic": "System Scalability",
                        "difficulty": "medium",
                        "target_skill": "conciseness",
                    },
                },
                "Clarity",
            ),
        ]:
            res = client.post(
                "/api/v1/voice-practice/session-ticket",
                headers=headers,
                json=mode_payload,
            )
            assert res.status_code == 201
            s_id = res.json()["data"]["session_id"]
            t_val = res.json()["data"]["ticket"]

            with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={t_val}") as ws:
                _ = ws.receive_json()  # session_ready
                while True:
                    evt = ws.receive_json()
                    if evt["event"] == "turn_completed":
                        break
                ws.send_json({"event": "session_end"})
                _ = ws.receive_json()

            eval_res = client.post(
                f"/api/v1/voice-practice/sessions/{s_id}/evaluate",
                headers=headers,
            )
            assert eval_res.status_code == 200
            fb = eval_res.json()["data"]
            assert fb["evaluation_status"] == "completed"
            assert fb["overall_score"] == 86.5
            assert any(d["dimension"] == expected_dim for d in fb["dimension_scores"])
            assert fb["source_references"] == []


def test_phase5_backend_safeguards_grounding_status_pending_reconciliation_and_atomic_eval_claim(
    phase4_app_factory: Any,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = phase4_app_factory()
    with TestClient(app) as client:
        headers, _, user_id = _register_and_get_headers(client, "safeguards.p5@example.com")
        ready_doc, _ = _seed_user_document_with_chunks(
            db_session,
            user_id=user_id,
            filename="safeguards_doc.pdf",
            processing_status=ProcessingStatus.READY.value,
        )

        # 1. Degraded RAG retrieval returns grounding_status="degraded_retrieval_error"
        def _broken_retrieve(*_args: Any, **_kwargs: Any) -> list[RetrievedChunk]:
            raise RuntimeError("Embedding provider timeout")

        monkeypatch.setattr(RetrievalService, "retrieve_relevant_chunks", _broken_retrieve)

        ticket_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
            json={
                "mode": "document_oral",
                "document_oral_config": {
                    "document_ids": [str(ready_doc.id)],
                    "difficulty": "medium",
                    "question_count": 3,
                },
            },
        )
        assert ticket_res.status_code == 201
        ticket_data = ticket_res.json()["data"]
        assert ticket_data["grounding_status"] == "degraded_retrieval_error"
        pending_session_id = uuid.UUID(ticket_data["session_id"])

        # 2. Expired pending session reconciliation transitions stale pending sessions to failed
        repo = VoicePracticeRepository(db_session)
        stale_session = repo.get_session_for_user(pending_session_id, user_id)
        assert stale_session is not None
        assert stale_session.status == "pending"
        stale_session.created_at = datetime.now(UTC) - timedelta(minutes=5)
        db_session.commit()

        detail_res = client.get(
            f"/api/v1/voice-practice/sessions/{pending_session_id}",
            headers=headers,
        )
        assert detail_res.status_code == 200
        detail_data = detail_res.json()["data"]
        assert detail_data["status"] == "failed"
        assert detail_data["termination_reason"] == "ticket_expired"
        assert detail_data["grounding_status"] == "degraded_retrieval_error"

        # 3. Atomic claim_session_for_evaluation prevents concurrent claims
        active_session_id = uuid.uuid4()
        repo.create_session(
            session_id=active_session_id,
            user_id=user_id,
            mode="interview",
            title="Concurrent Claim Test",
            config_json={},
            document_ids=[],
            retrieved_chunks=[],
        )
        repo.finalize_session(
            active_session_id,
            status="completed",
            ended_at=datetime.now(UTC),
            duration_seconds=45,
            termination_reason="client_ended",
        )
        db_session.commit()

        assert (
            repo.claim_session_for_evaluation(
                active_session_id,
                user_id,
                model_name="gemini-2.5-flash",
            )
            is True
        )
        db_session.commit()
        assert (
            repo.claim_session_for_evaluation(
                active_session_id,
                user_id,
                model_name="gemini-2.5-flash",
            )
            is False
        )
