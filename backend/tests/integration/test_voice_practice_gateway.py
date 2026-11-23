import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google import genai
from google.genai import types
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.main import create_app
from app.modules.auth.models import AccountStatus, User
from app.modules.chat.models import ChatConversation, ChatMessage
from app.modules.voice_practice.gateway import (
    GeminiLiveClientFactory,
    GeminiLiveSessionProtocol,
    get_gemini_live_client_factory,
)
from app.modules.voice_practice.schemas import ActiveVoiceSession
from app.modules.voice_practice.service import (
    InMemoryVoiceSessionStore,
    get_voice_session_store,
    hash_connection_ticket,
)


class FakeGeminiLiveSession:
    """Controllable fake Gemini Live session for deterministic WebSocket gateway testing."""

    def __init__(
        self,
        *,
        turns: list[list[Any]] | None = None,
        hold_open_after_turns: bool = True,
        wait_for_audio_end: bool = False,
        raise_on_receive: Exception | None = None,
        raise_on_send_audio: Exception | None = None,
    ) -> None:
        self.sent_audio_frames: list[tuple[bytes, int]] = []
        self.audio_stream_end_count: int = 0
        self.closed: bool = False
        self._turns = list(turns) if turns is not None else []
        self._hold_open = hold_open_after_turns
        self._wait_for_audio_end = wait_for_audio_end
        self._raise_on_receive = raise_on_receive
        self._raise_on_send_audio = raise_on_send_audio
        self._turn_cursor = 0
        self._audio_end_event = asyncio.Event()

    async def send_audio(self, pcm_bytes: bytes, *, sample_rate_hz: int) -> None:
        if self._raise_on_send_audio is not None:
            raise self._raise_on_send_audio
        self.sent_audio_frames.append((pcm_bytes, sample_rate_hz))

    async def send_audio_stream_end(self) -> None:
        self.audio_stream_end_count += 1
        self._audio_end_event.set()

    async def receive(self) -> AsyncIterator[Any]:
        if self._raise_on_receive is not None:
            raise self._raise_on_receive

        if self._wait_for_audio_end and self._turn_cursor == 0:
            await self._audio_end_event.wait()

        if self._turn_cursor < len(self._turns):
            current_turn = self._turns[self._turn_cursor]
            self._turn_cursor += 1
            completed_turn = False
            for msg in current_turn:
                sc = getattr(msg, "server_content", None)
                if sc is not None and getattr(sc, "turn_complete", False):
                    completed_turn = True
                yield msg
            if completed_turn or not self._hold_open:
                return

        if self._hold_open:
            while not self.closed:
                await asyncio.sleep(0.02)

    async def close(self) -> None:
        self.closed = True


def _make_server_message(
    *,
    user_transcript: str | None = None,
    user_finished: bool | None = None,
    assistant_transcript: str | None = None,
    assistant_finished: bool | None = None,
    audio_bytes: bytes | None = None,
    interrupted: bool | None = None,
    turn_complete: bool | None = None,
) -> types.LiveServerMessage:
    parts: list[types.Part] = []
    if audio_bytes is not None:
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    data=audio_bytes,
                    mime_type="audio/pcm;rate=24000",
                )
            )
        )
    model_turn = types.Content(role="model", parts=parts) if parts else None
    input_tx = (
        types.Transcription(text=user_transcript, finished=user_finished)
        if user_transcript is not None or user_finished is not None
        else None
    )
    output_tx = (
        types.Transcription(text=assistant_transcript, finished=assistant_finished)
        if assistant_transcript is not None or assistant_finished is not None
        else None
    )
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=model_turn,
            input_transcription=input_tx,
            output_transcription=output_tx,
            interrupted=interrupted,
            turn_complete=turn_complete,
        )
    )


def _register_and_get_headers(
    client: TestClient,
    email: str = "voice.student@example.com",
) -> tuple[dict[str, str], str, uuid.UUID]:
    res = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPassword123!",
        },
    )
    assert res.status_code == 201
    body = res.json()["data"]
    token = body["tokens"]["access_token"]
    user_id = uuid.UUID(body["user"]["id"])
    return {"Authorization": f"Bearer {token}"}, token, user_id


@pytest.fixture
def voice_store() -> InMemoryVoiceSessionStore:
    return InMemoryVoiceSessionStore()


@pytest.fixture
def voice_settings() -> Settings:
    return Settings(
        APP_ENV="testing",
        GEMINI_API_KEY="mock-gemini-api-key-never-log-me-9999",
        GEMINI_LIVE_MODEL="gemini-2.5-flash-native-audio-latest",
        VOICE_TICKET_TTL_SECONDS=60,
        VOICE_MAX_SESSION_DURATION_SECONDS=600,
        VOICE_SESSION_WARNING_SECONDS=570,
        VOICE_MAX_CONCURRENT_SESSIONS_PER_USER=1,
        VOICE_MAX_GLOBAL_CONCURRENT_SESSIONS=3,
        VOICE_MAX_DAILY_SESSIONS_PER_USER=3,
        VOICE_MAX_AUDIO_FRAME_BYTES=4096,
        VOICE_MAX_JSON_MESSAGE_BYTES=2048,
    )


@pytest.fixture
def voice_app_factory(
    db_session: Session,
    voice_store: InMemoryVoiceSessionStore,
    voice_settings: Settings,
) -> Any:
    _ = db_session

    def _build(
        *,
        live_factory: GeminiLiveClientFactory | None = None,
        custom_settings: Settings | None = None,
        custom_store: InMemoryVoiceSessionStore | None = None,
    ) -> Any:
        active_settings = custom_settings or voice_settings
        active_store = custom_store or voice_store
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: active_settings
        app.dependency_overrides[get_voice_session_store] = lambda: active_store

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

            app.dependency_overrides[get_gemini_live_client_factory] = (
                lambda: _default_fake_factory
            )
        return app

    return _build


# ============================================================================
# 1. AUTHENTICATION & TICKET TESTS
# ============================================================================


def test_ticket_issuance_requires_authentication(voice_app_factory: Any) -> None:
    app = voice_app_factory()
    with TestClient(app) as client:
        res = client.post("/api/v1/voice-practice/session-ticket")
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"


def test_valid_ticket_single_use_and_reused_ticket_rejected(voice_app_factory: Any) -> None:
    app = voice_app_factory()
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client)
        ticket_res = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
            json={"mode": "neutral", "input_sample_rate_hz": 16000},
        )
        assert ticket_res.status_code == 201
        data = ticket_res.json()["data"]
        ticket = data["ticket"]
        assert data["expires_in_seconds"] == 60
        assert data["audio_format"]["input_sample_rate_hz"] == 16000
        assert data["audio_format"]["output_sample_rate_hz"] == 24000
        assert data["limits"]["max_duration_seconds"] == 600
        assert data["limits"]["warning_at_seconds"] == 570

        # 1st connection with valid ticket succeeds
        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            ready = ws.receive_json()
            assert ready["event"] == "session_ready"
            assert ready["payload"]["model"] == "gemini-2.5-flash-native-audio-latest"
            ws.send_json({"event": "session_end"})
            ended = ws.receive_json()
            assert ended["event"] == "session_ended"
            assert ended["payload"]["reason"] == "client_requested"

        # 2nd connection reusing the same ticket is rejected
        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws2:
            err = ws2.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "INVALID_CONNECTION_TICKET"
            assert err["payload"]["fatal"] is True


def test_missing_invalid_expired_and_jwt_tickets_rejected(
    voice_app_factory: Any,
    voice_settings: Settings,
) -> None:
    current_clock = datetime(2026, 11, 7, 10, 0, 0, tzinfo=UTC)
    clock_store = InMemoryVoiceSessionStore(now_fn=lambda: current_clock)
    app = voice_app_factory(custom_store=clock_store)

    with TestClient(app) as client:
        headers, access_token, _ = _register_and_get_headers(client, "ticket.edge@example.com")

        # 1. Missing ticket
        with client.websocket_connect("/api/v1/voice-practice/ws") as ws:
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "MISSING_CONNECTION_TICKET"

        # 2. Invalid ticket
        with client.websocket_connect(
            "/api/v1/voice-practice/ws?ticket=completely-invalid-random-ticket-value-12345"
        ) as ws:
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "INVALID_CONNECTION_TICKET"

        # 3. Forbidden query parameter (access_token or api_key in query string)
        with client.websocket_connect(
            f"/api/v1/voice-practice/ws?access_token={access_token}"
        ) as ws:
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "FORBIDDEN_QUERY_PARAMETER"

        # 4. Passing a JWT directly as `ticket` parameter is rejected
        with client.websocket_connect(
            f"/api/v1/voice-practice/ws?ticket={access_token}"
        ) as ws:
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "JWT_NOT_ALLOWED_IN_WS"

        # 5. Expired ticket
        ticket_res = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
        assert ticket_res.status_code == 201
        raw_ticket = ticket_res.json()["data"]["ticket"]
        # Force expiry in store
        t_hash = hash_connection_ticket(raw_ticket)
        clock_store._tickets[t_hash].expires_at = current_clock - timedelta(seconds=1)

        with client.websocket_connect(
            f"/api/v1/voice-practice/ws?ticket={raw_ticket}"
        ) as ws:
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "INVALID_CONNECTION_TICKET"
            assert clock_store.get_global_active_count() == 0
    _ = voice_settings


def test_inactive_user_cannot_connect_with_issued_ticket(
    voice_app_factory: Any,
    db_session: Session,
) -> None:
    app = voice_app_factory()
    with TestClient(app) as client:
        headers, _, user_id = _register_and_get_headers(client, "suspend.me@example.com")
        ticket_res = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
        assert ticket_res.status_code == 201
        ticket = ticket_res.json()["data"]["ticket"]

        # Suspend user before WebSocket connection
        user = db_session.get(User, user_id)
        assert user is not None
        user.status = AccountStatus.SUSPENDED.value
        db_session.commit()

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "ACCOUNT_INACTIVE"


# ============================================================================
# 2. STREAMING, TRANSCRIPTS, VAD INTERRUPTION & TURN COMPLETION TESTS
# ============================================================================


def test_bidirectional_audio_streaming_transcripts_and_turn_completion(
    voice_app_factory: Any,
) -> None:
    pcm_out_chunk_1 = b"\x10\x00" * 240
    pcm_out_chunk_2 = b"\x20\x00" * 240

    turn_1_messages = [
        _make_server_message(user_transcript="What is ", user_finished=False),
        _make_server_message(user_transcript="mitosis?", user_finished=True),
        _make_server_message(
            audio_bytes=pcm_out_chunk_1,
            assistant_transcript="Mitosis is ",
            assistant_finished=False,
        ),
        _make_server_message(
            audio_bytes=pcm_out_chunk_2,
            assistant_transcript="cell division.",
            assistant_finished=True,
            turn_complete=True,
        ),
    ]

    fake_session = FakeGeminiLiveSession(
        turns=[turn_1_messages],
        hold_open_after_turns=True,
        wait_for_audio_end=True,
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

    app = voice_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "stream.user@example.com")
        ticket = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            ready = ws.receive_json()
            assert ready["event"] == "session_ready"

            # Send client control + binary PCM audio input
            ws.send_json({"event": "audio_input_started", "payload": {"sample_rate_hz": 16000}})
            pcm_in = b"\x05\x00" * 320  # 640 bytes (even, 16-bit PCM)
            ws.send_bytes(pcm_in)
            ws.send_json({"event": "audio_input_stopped"})

            events: list[dict[str, Any]] = []
            received_audio_frames: list[bytes] = []

            while True:
                raw_msg = ws.receive()
                if raw_msg.get("bytes") is not None:
                    received_audio_frames.append(raw_msg["bytes"])
                elif raw_msg.get("text") is not None:
                    import json as _json

                    evt = _json.loads(raw_msg["text"])
                    events.append(evt)
                    if evt["event"] == "turn_completed":
                        break

            ws.send_json({"event": "session_end"})
            end_evt = ws.receive_json()
            assert end_evt["event"] == "session_ended"

        # Verify audio input was forwarded to Gemini Live
        assert len(fake_session.sent_audio_frames) == 1
        assert fake_session.sent_audio_frames[0] == (pcm_in, 16000)
        assert fake_session.audio_stream_end_count == 1
        assert fake_session.closed is True

        # Verify binary audio output frames arrived intact and in order
        assert received_audio_frames == [pcm_out_chunk_1, pcm_out_chunk_2]

        # Verify event sequence and transcripts
        event_names = [e["event"] for e in events]
        assert "session_started" in event_names
        assert "audio_input_started" in event_names
        assert "audio_input_stopped" in event_names
        assert "audio_output_started" in event_names
        assert "audio_output_finished" in event_names
        assert "turn_completed" in event_names

        user_finals = [
            e["payload"]["text"]
            for e in events
            if e["event"] == "transcript_final" and e["payload"]["role"] == "user"
        ]
        assert user_finals == ["What is mitosis?"]

        assistant_finals = [
            e["payload"]["text"]
            for e in events
            if e["event"] == "transcript_final" and e["payload"]["role"] == "assistant"
        ]
        assert assistant_finals == ["Mitosis is cell division."]


def test_interruption_discards_stale_audio_and_notifies_client(
    voice_app_factory: Any,
) -> None:
    valid_audio = b"\x11\x00" * 100
    stale_audio_after_interrupt = b"\x99\x00" * 100

    interrupted_turn = [
        _make_server_message(
            audio_bytes=valid_audio,
            assistant_transcript="Let me explain a very long ",
        ),
        _make_server_message(interrupted=True),
        # Stale audio chunk arriving before turn_complete must be discarded
        _make_server_message(
            audio_bytes=stale_audio_after_interrupt,
            assistant_transcript="stale text that should be ignored",
        ),
        _make_server_message(turn_complete=True),
    ]

    fake_session = FakeGeminiLiveSession(turns=[interrupted_turn], hold_open_after_turns=True)

    @asynccontextmanager
    async def _factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        yield fake_session

    app = voice_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "interrupt.user@example.com")
        ticket = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            _ = ws.receive_json()  # session_ready
            events: list[dict[str, Any]] = []
            received_audio: list[bytes] = []

            while True:
                raw_msg = ws.receive()
                if raw_msg.get("bytes") is not None:
                    received_audio.append(raw_msg["bytes"])
                elif raw_msg.get("text") is not None:
                    import json as _json

                    evt = _json.loads(raw_msg["text"])
                    events.append(evt)
                    if evt["event"] == "turn_completed":
                        break

            ws.send_json({"event": "session_end"})
            _ = ws.receive_json()

        # Stale audio after `interrupted=True` must NEVER be sent to the client
        assert stale_audio_after_interrupt not in received_audio

        interrupted_events = [e for e in events if e["event"] == "interrupted"]
        assert len(interrupted_events) == 1
        assert interrupted_events[0]["payload"]["source"] == "user_speech"

        finished_events = [e for e in events if e["event"] == "audio_output_finished"]
        assert len(finished_events) == 1
        assert finished_events[0]["payload"]["reason"] == "interrupted"

        turn_completed = [e for e in events if e["event"] == "turn_completed"][0]
        assert turn_completed["payload"]["interrupted"] is True


def test_malformed_json_and_invalid_audio_frames_handled_safely(
    voice_app_factory: Any,
) -> None:
    fake_session = FakeGeminiLiveSession(turns=[], hold_open_after_turns=True)

    @asynccontextmanager
    async def _factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        yield fake_session

    app = voice_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "validation.user@example.com")
        ticket = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            ready = ws.receive_json()
            assert ready["event"] == "session_ready"

            # 1. Malformed JSON string
            ws.send_text("{not-valid-json")
            err1 = ws.receive_json()
            assert err1["event"] == "session_error"
            assert err1["payload"]["code"] == "INVALID_CONTROL_MESSAGE"
            assert err1["payload"]["fatal"] is False

            # 2. Odd-length binary PCM frame (invalid 16-bit PCM)
            ws.send_bytes(b"\x01\x02\x03")
            err2 = ws.receive_json()
            assert err2["event"] == "session_error"
            assert err2["payload"]["code"] == "INVALID_AUDIO_FRAME"
            assert err2["payload"]["fatal"] is False

            # 3. Oversized binary frame (> 4096 bytes in test settings)
            ws.send_bytes(b"\x00\x00" * 3000)
            err3 = ws.receive_json()
            assert err3["event"] == "session_error"
            assert err3["payload"]["code"] == "AUDIO_FRAME_TOO_LARGE"
            assert err3["payload"]["fatal"] is False

            # Ensure none of the invalid frames were forwarded to Gemini
            assert len(fake_session.sent_audio_frames) == 0

            # Connection remains healthy for valid ping/pong and clean close
            ws.send_json({"event": "ping", "payload": {"timestamp": 12345}})
            pong = ws.receive_json()
            assert pong["event"] == "pong"
            assert pong["payload"]["client_timestamp"] == 12345

            ws.send_json({"event": "session_end"})
            ended = ws.receive_json()
            assert ended["event"] == "session_ended"


# ============================================================================
# 3. SESSION LIFECYCLE, TIMEOUT, CONCURRENCY & QUOTA TESTS
# ============================================================================


def test_session_warning_and_authoritative_max_duration_timeout(
    voice_app_factory: Any,
    voice_settings: Settings,
) -> None:
    # Configure fast timer thresholds for deterministic testing (warning at 1s, max at 2s)
    short_timer_settings = voice_settings.model_copy(
        update={
            "VOICE_SESSION_WARNING_SECONDS": 1,
            "VOICE_MAX_SESSION_DURATION_SECONDS": 2,
        }
    )
    app = voice_app_factory(custom_settings=short_timer_settings)

    with TestClient(app) as client:
        headers, _, _ = _register_and_get_headers(client, "timeout.user@example.com")
        ticket = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            ready = ws.receive_json()
            assert ready["event"] == "session_ready"

            warning = ws.receive_json()
            assert warning["event"] == "session_warning"
            assert warning["payload"]["remaining_seconds"] >= 1

            ended = ws.receive_json()
            assert ended["event"] == "session_ended"
            assert ended["payload"]["reason"] == "max_duration_reached"


def test_gemini_initialization_failure_releases_lock_and_refunds_daily_quota(
    voice_app_factory: Any,
    voice_store: InMemoryVoiceSessionStore,
) -> None:
    @asynccontextmanager
    async def _failing_factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        raise RuntimeError("Simulated upstream Gemini handshake failure")
        yield  # pragma: no cover

    app = voice_app_factory(live_factory=_failing_factory)
    with TestClient(app) as client:
        headers, _, user_id = _register_and_get_headers(client, "init.fail@example.com")
        ticket = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            err = ws.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "VOICE_SESSION_INIT_FAILED"
            assert err["payload"]["fatal"] is True

            ended = ws.receive_json()
            assert ended["event"] == "session_ended"
            assert ended["payload"]["reason"] == "error"

        # Lock is released AND daily quota was refunded because Gemini never initialized
        assert voice_store.is_user_session_active(user_id) is False
        assert voice_store.get_global_active_count() == 0
        assert voice_store.get_daily_session_count(user_id) == 0


def test_gemini_disconnection_and_midstream_exception_cleanup(
    voice_app_factory: Any,
    voice_store: InMemoryVoiceSessionStore,
) -> None:
    # Case A: Gemini closes stream cleanly without further turns (provider_closed)
    closing_session = FakeGeminiLiveSession(turns=[], hold_open_after_turns=False)

    @asynccontextmanager
    async def _closing_factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        try:
            yield closing_session
        finally:
            await closing_session.close()

    app1 = voice_app_factory(live_factory=_closing_factory)
    with TestClient(app1) as client:
        headers, _, user_id = _register_and_get_headers(client, "gemini.close@example.com")
        t1 = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]
        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={t1}") as ws:
            ready = ws.receive_json()
            assert ready["event"] == "session_ready"
            ended = ws.receive_json()
            assert ended["event"] == "session_ended"
            assert ended["payload"]["reason"] == "provider_closed"

        assert closing_session.closed is True
        assert voice_store.is_user_session_active(user_id) is False
        assert voice_store.get_global_active_count() == 0

        # Case B: Gemini raises an unexpected exception mid-stream -> GEMINI_STREAM_CLOSED + cleanup
        crashing_session = FakeGeminiLiveSession(
            raise_on_receive=ConnectionResetError("Upstream socket reset"),
        )

        @asynccontextmanager
        async def _crashing_factory(
            _session: ActiveVoiceSession,
            _cfg: Settings,
        ) -> AsyncIterator[GeminiLiveSessionProtocol]:
            try:
                yield crashing_session
            finally:
                await crashing_session.close()

        app1.dependency_overrides[get_gemini_live_client_factory] = lambda: _crashing_factory
        t2 = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]
        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={t2}") as ws2:
            _ = ws2.receive_json()  # session_ready
            err = ws2.receive_json()
            assert err["event"] == "session_error"
            assert err["payload"]["code"] == "GEMINI_STREAM_CLOSED"
            ended2 = ws2.receive_json()
            assert ended2["event"] == "session_ended"
            assert ended2["payload"]["reason"] == "error"

        assert crashing_session.closed is True
        assert voice_store.is_user_session_active(user_id) is False
        assert voice_store.get_global_active_count() == 0


def test_client_abrupt_disconnection_releases_resources(
    voice_app_factory: Any,
    voice_store: InMemoryVoiceSessionStore,
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

    app = voice_app_factory(live_factory=_factory)
    with TestClient(app) as client:
        headers, _, user_id = _register_and_get_headers(client, "abrupt.disconnect@example.com")
        ticket = client.post(
            "/api/v1/voice-practice/session-ticket",
            headers=headers,
        ).json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={ticket}") as ws:
            ready = ws.receive_json()
            assert ready["event"] == "session_ready"
            assert voice_store.is_user_session_active(user_id) is True
            # Exit context manager abruptly without sending session_end

        assert fake_session.closed is True
        assert voice_store.is_user_session_active(user_id) is False
        assert voice_store.get_global_active_count() == 0


def test_per_user_concurrency_and_daily_quota_enforcement(
    voice_app_factory: Any,
    voice_store: InMemoryVoiceSessionStore,
) -> None:
    app = voice_app_factory()
    with TestClient(app) as client:
        headers, _, user_id = _register_and_get_headers(client, "quota.user@example.com")

        # Issue two tickets before connecting, then verify second WS connect is blocked
        # while first session is active
        res1 = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
        res2 = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
        t1 = res1.json()["data"]["ticket"]
        t2 = res2.json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={t1}") as ws1:
            ready = ws1.receive_json()
            assert ready["event"] == "session_ready"
            assert voice_store.is_user_session_active(user_id) is True

            # Requesting a new ticket while session is active fails with 409 CONFLICT
            conflict_res = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
            assert conflict_res.status_code == 409
            assert conflict_res.json()["error"]["code"] == "VOICE_SESSION_ALREADY_ACTIVE"

            # Connecting with pre-issued ticket t2 while ws1 is active fails
            with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={t2}") as ws2:
                err = ws2.receive_json()
                assert err["event"] == "session_error"
                assert err["payload"]["code"] == "VOICE_SESSION_ALREADY_ACTIVE"

            ws1.send_json({"event": "session_end"})
            _ = ws1.receive_json()

        # After ws1 closes, lock is released and daily count is 1
        assert voice_store.is_user_session_active(user_id) is False
        assert voice_store.get_daily_session_count(user_id) == 1

        # Complete sessions 2 and 3 for the day
        for expected_remaining in (2, 1):
            res = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
            assert res.status_code == 201
            assert res.json()["data"]["limits"]["daily_quota_remaining"] == expected_remaining
            tx = res.json()["data"]["ticket"]
            with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={tx}") as wsx:
                _ = wsx.receive_json()
                wsx.send_json({"event": "session_end"})
                _ = wsx.receive_json()

        assert voice_store.get_daily_session_count(user_id) == 3

        # 4th session attempt on the same day is rejected with HTTP 429 VOICE_DAILY_QUOTA_EXCEEDED
        quota_res = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
        assert quota_res.status_code == 429
        assert quota_res.json()["error"]["code"] == "VOICE_DAILY_QUOTA_EXCEEDED"


def test_global_concurrency_cap_of_three_sessions(
    voice_app_factory: Any,
    voice_store: InMemoryVoiceSessionStore,
) -> None:
    app = voice_app_factory()
    with TestClient(app) as client:
        users = [
            _register_and_get_headers(client, f"global.{idx}@example.com")
            for idx in range(4)
        ]

        tickets = [
            client.post(
                "/api/v1/voice-practice/session-ticket",
                headers=u[0],
            ).json()["data"]["ticket"]
            for u in users[:3]
        ]

        with (
            client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={tickets[0]}") as ws0,
            client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={tickets[1]}") as ws1,
            client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={tickets[2]}") as ws2,
        ):
            assert ws0.receive_json()["event"] == "session_ready"
            assert ws1.receive_json()["event"] == "session_ready"
            assert ws2.receive_json()["event"] == "session_ready"
            assert voice_store.get_global_active_count() == 3

            # 4th user requesting a ticket while 3 global sessions are active receives 503
            res_u4 = client.post("/api/v1/voice-practice/session-ticket", headers=users[3][0])
            assert res_u4.status_code == 503
            assert res_u4.json()["error"]["code"] == "VOICE_GLOBAL_CAPACITY_REACHED"

            ws0.send_json({"event": "session_end"})
            ws1.send_json({"event": "session_end"})
            ws2.send_json({"event": "session_end"})
            _ = ws0.receive_json()
            _ = ws1.receive_json()
            _ = ws2.receive_json()

        assert voice_store.get_global_active_count() == 0


# ============================================================================
# 4. PRIVACY & SECURITY VERIFICATION TESTS
# ============================================================================


def test_privacy_no_raw_audio_on_disk_no_secrets_logged_and_no_db_transcripts(
    voice_app_factory: Any,
    voice_settings: Settings,
    db_session: Session,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_audio_marker = b"SECRET_PCM_AUDIO_BYTES_DO_NOT_LOG_OR_PERSIST!!"  # 46 bytes (even)
    turn_messages = [
        _make_server_message(
            user_transcript="Confidential spoken practice question",
            user_finished=True,
            audio_bytes=secret_audio_marker,
            assistant_transcript="Confidential spoken practice answer",
            assistant_finished=True,
            turn_complete=True,
        )
    ]
    fake_session = FakeGeminiLiveSession(turns=[turn_messages], hold_open_after_turns=True)

    @asynccontextmanager
    async def _factory(
        _session: ActiveVoiceSession,
        _cfg: Settings,
    ) -> AsyncIterator[GeminiLiveSessionProtocol]:
        yield fake_session

    app = voice_app_factory(live_factory=_factory)

    with caplog.at_level(logging.DEBUG), TestClient(app) as client:
        headers, access_token, _ = _register_and_get_headers(client, "privacy.user@example.com")
        ticket_res = client.post("/api/v1/voice-practice/session-ticket", headers=headers)
        raw_ticket = ticket_res.json()["data"]["ticket"]

        with client.websocket_connect(f"/api/v1/voice-practice/ws?ticket={raw_ticket}") as ws:
            _ = ws.receive_json()  # session_ready
            ws.send_bytes(secret_audio_marker)
            while True:
                msg = ws.receive()
                if msg.get("text"):
                    import json as _json

                    evt = _json.loads(msg["text"])
                    if evt["event"] == "turn_completed":
                        break
            ws.send_json({"event": "session_end"})
            _ = ws.receive_json()

    # 1. Verify no credentials, JWTs, tickets, or audio payloads appear in logs
    combined_logs = "\n".join(caplog.messages)
    assert (voice_settings.GEMINI_API_KEY or "") not in combined_logs
    assert access_token not in combined_logs
    assert raw_ticket not in combined_logs
    assert "SECRET_PCM_AUDIO_BYTES" not in combined_logs

    # 2. Verify no transcripts were persisted to PostgreSQL in Phase 3
    chat_conv_count = db_session.scalar(select(func.count()).select_from(ChatConversation))
    chat_msg_count = db_session.scalar(select(func.count()).select_from(ChatMessage))
    assert chat_conv_count == 0
    assert chat_msg_count == 0

    # 3. Verify no audio files were written to tmp_path
    assert list(tmp_path.rglob("*.pcm")) == []
    assert list(tmp_path.rglob("*.wav")) == []
    assert list(tmp_path.rglob("*.m4a")) == []


# ============================================================================
# 5. OPT-IN REAL GEMINI LIVE INTEGRATION TEST
# ============================================================================


@pytest.mark.skipif(
    os.environ.get("RUN_GEMINI_LIVE_INTEGRATION") != "1",
    reason="Opt-in real Gemini Live API test. Set RUN_GEMINI_LIVE_INTEGRATION=1 to run.",
)
def test_opt_in_real_gemini_live_connection() -> None:
    settings = get_settings()
    assert settings.GEMINI_API_KEY, "GEMINI_API_KEY must be set for live integration test"

    async def _verify_live() -> None:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        cfg = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        async with client.aio.live.connect(
            model=settings.GEMINI_LIVE_MODEL,
            config=cfg,
        ) as session:
            await session.send_realtime_input(
                audio=types.Blob(data=b"\x00\x00" * 1600, mime_type="audio/pcm;rate=16000")
            )
            await session.send_realtime_input(audio_stream_end=True)

    asyncio.run(_verify_live())
