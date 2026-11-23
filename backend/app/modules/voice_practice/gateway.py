import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.core.config import Settings, get_settings
from app.modules.voice_practice.schemas import (
    DEFAULT_INPUT_SAMPLE_RATE_HZ,
    GEMINI_OUTPUT_SAMPLE_RATE_HZ,
    PCM_BYTES_PER_SAMPLE,
    SUPPORTED_INPUT_SAMPLE_RATES,
    ActiveVoiceSession,
    AudioInputStartedPayload,
    AudioInputStoppedPayload,
    AudioOutputFinishedPayload,
    AudioOutputStartedPayload,
    ClientControlMessage,
    ClientEventType,
    InterruptedPayload,
    ServerEventEnvelope,
    ServerEventType,
    SessionEndedPayload,
    SessionErrorPayload,
    SessionReadyPayload,
    SessionStartedPayload,
    SessionWarningPayload,
    TranscriptEventPayload,
    TurnCompletedPayload,
    VoiceAudioFormatSpec,
    VoicePracticeMode,
)
from app.modules.voice_practice.service import VoiceGatewayPersistence

logger = logging.getLogger("app.modules.voice_practice.gateway")

VOICE_PRACTICE_SYSTEM_INSTRUCTION = """\
You are StudyGuide Voice Practice Assistant, an encouraging, clear, and articulate academic \
study partner. Engage in natural spoken conversation to help the user practice explaining \
concepts, answering study questions, and improving their verbal clarity.
Keep your spoken responses concise, conversational, and well-structured for real-time audio. \
Allow the user to interrupt naturally at any time.
"""


def normalize_pcm16le_to_16khz(pcm_bytes: bytes, source_rate_hz: int) -> bytes:
    """
    Normalizes 16-bit signed little-endian mono PCM audio frames to Gemini Live's canonical
    16,000 Hz sample rate (`audio/pcm;rate=16000`).

    - Rejects any sample rate not in `SUPPORTED_INPUT_SAMPLE_RATES`.
    - Passes 16,000 Hz audio through zero-copy.
    - Resamples 24,000 Hz, 44,100 Hz, and 48,000 Hz frames to 16,000 Hz via linear interpolation.
    """
    if (
        isinstance(source_rate_hz, bool)
        or not isinstance(source_rate_hz, int)
        or source_rate_hz not in SUPPORTED_INPUT_SAMPLE_RATES
    ):
        raise ValueError(
            f"Unsupported input sample rate {source_rate_hz} Hz. "
            f"Supported rates: {SUPPORTED_INPUT_SAMPLE_RATES}"
        )

    if len(pcm_bytes) % PCM_BYTES_PER_SAMPLE != 0:
        raise ValueError("PCM 16-bit audio frame must have an even byte length")

    if source_rate_hz == DEFAULT_INPUT_SAMPLE_RATE_HZ or len(pcm_bytes) == 0:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float64)
    in_len = int(samples.shape[0])
    if in_len <= 1:
        return pcm_bytes

    out_len = max(1, int(round(in_len * DEFAULT_INPUT_SAMPLE_RATE_HZ / source_rate_hz)))
    x_old = np.linspace(0.0, 1.0, num=in_len, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
    resampled = np.interp(x_new, x_old, samples)
    clipped = np.clip(np.round(resampled), -32768, 32767).astype("<i2")
    return clipped.tobytes()


class GeminiLiveSessionProtocol(Protocol):
    """Protocol abstracting an active Gemini Live bidirectional session."""

    async def send_audio(self, pcm_bytes: bytes, *, sample_rate_hz: int) -> None: ...

    async def send_audio_stream_end(self) -> None: ...

    def receive(self) -> AsyncIterator[Any]: ...

    async def close(self) -> None: ...


GeminiLiveClientFactory = Callable[
    [ActiveVoiceSession, Settings],
    AbstractAsyncContextManager[GeminiLiveSessionProtocol],
]


class GoogleGenAILiveSession:
    """Adapter wrapping `google.genai.live.AsyncSession` for real-time PCM audio streaming."""

    def __init__(self, raw_session: Any) -> None:
        self._session = raw_session

    async def send_audio(self, pcm_bytes: bytes, *, sample_rate_hz: int) -> None:
        normalized_pcm = normalize_pcm16le_to_16khz(pcm_bytes, sample_rate_hz)
        mime_type = f"audio/pcm;rate={DEFAULT_INPUT_SAMPLE_RATE_HZ}"
        await self._session.send_realtime_input(
            audio=types.Blob(data=normalized_pcm, mime_type=mime_type),
        )

    async def send_audio_stream_end(self) -> None:
        await self._session.send_realtime_input(audio_stream_end=True)

    def receive(self) -> AsyncIterator[Any]:
        result: AsyncIterator[Any] = self._session.receive()
        return result

    async def close(self) -> None:
        close_fn = getattr(self._session, "close", None)
        if callable(close_fn):
            res = close_fn()
            if asyncio.iscoroutine(res):
                await res


def build_live_connect_config(
    session: ActiveVoiceSession,
    settings: Settings,
) -> types.LiveConnectConfig:
    effective_instruction = (
        session.system_instruction.strip()
        if session.system_instruction and session.system_instruction.strip()
        else VOICE_PRACTICE_SYSTEM_INSTRUCTION
    )
    silence_duration_ms = (
        settings.VOICE_INTERVIEW_SILENCE_DURATION_MS
        if session.mode == VoicePracticeMode.INTERVIEW
        else settings.VOICE_CONVERSATIONAL_SILENCE_DURATION_MS
    )
    end_sensitivity = (
        types.EndSensitivity.END_SENSITIVITY_HIGH
        if settings.VOICE_VAD_END_SENSITIVITY == "high"
        else types.EndSensitivity.END_SENSITIVITY_LOW
    )
    start_sensitivity = (
        types.StartSensitivity.START_SENSITIVITY_LOW
        if settings.VOICE_VAD_START_SENSITIVITY == "low"
        else types.StartSensitivity.START_SENSITIVITY_HIGH
    )
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=start_sensitivity,
                end_of_speech_sensitivity=end_sensitivity,
                prefix_padding_ms=settings.VOICE_VAD_PREFIX_PADDING_MS,
                silence_duration_ms=silence_duration_ms,
            ),
        ),
        system_instruction=effective_instruction,
    )


@asynccontextmanager
async def default_gemini_live_client_factory(
    session: ActiveVoiceSession,
    settings: Settings,
) -> AsyncIterator[GeminiLiveSessionProtocol]:
    """Creates and manages an official `google-genai` `client.aio.live.connect` session."""
    api_key = (settings.GEMINI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)
    live_config = build_live_connect_config(session, settings)
    async with client.aio.live.connect(
        model=settings.GEMINI_LIVE_MODEL,
        config=live_config,
    ) as raw_session:
        wrapped = GoogleGenAILiveSession(raw_session)
        try:
            yield wrapped
        finally:
            with contextlib.suppress(Exception):
                await wrapped.close()


def get_gemini_live_client_factory() -> GeminiLiveClientFactory:
    return default_gemini_live_client_factory


class VoicePracticeGateway:
    """
    Manages a single authenticated bidirectional WebSocket session between the mobile client
    and Google Gemini Live.

    Responsibilities:
    - Initialize Gemini Live connection and emit `session_ready`.
    - Validate and forward binary PCM audio input frames (`pcm_s16le`) without disk storage.
    - Stream binary PCM audio output frames (`24kHz pcm_s16le`) and JSON transcript events.
    - Coordinate server-side VAD interruptions and turn completion.
    - Enforce authoritative 9m30s warning and 10m00s maximum session duration.
    - Guarantee deterministic cleanup of tasks, buffers, and connections on exit.
    """

    def __init__(
        self,
        *,
        websocket: WebSocket,
        session: ActiveVoiceSession,
        settings: Settings | None = None,
        live_client_factory: GeminiLiveClientFactory | None = None,
        persistence: VoiceGatewayPersistence | None = None,
    ) -> None:
        self.websocket = websocket
        self.session = session
        self.settings = settings or get_settings()
        self.live_client_factory = live_client_factory or default_gemini_live_client_factory
        self.persistence = persistence or VoiceGatewayPersistence()

        self._send_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._start_monotonic = time.monotonic()

        self._input_sample_rate_hz: int = session.input_sample_rate_hz
        self._session_started_emitted: bool = False
        self._turn_index: int = 1
        self._persisted_turn_counter: int = 0
        self._audio_output_active: bool = False
        self._current_turn_interrupted: bool = False
        self._had_any_interruption: bool = False
        self._audio_stream_end_sent_for_turn: bool = False

        # Short-lived in-memory turn transcript accumulators (cleared each turn and on exit)
        self._user_transcript_buffer: list[str] = []
        self._user_transcript_finalized: bool = False
        self._user_turn_started_at: datetime | None = None
        self._user_persisted_turn_idx: int | None = None

        self._assistant_transcript_buffer: list[str] = []
        self._assistant_transcript_finalized: bool = False
        self._assistant_turn_started_at: datetime | None = None
        self._assistant_persisted_turn_idx: int | None = None

        # Outbound audio queue so queued frames can be flushed immediately on interruption
        self._outbound_audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=128)

        self._end_reason: Literal[
            "client_requested",
            "client_disconnected",
            "max_duration_reached",
            "provider_closed",
            "error",
        ] = "client_disconnected"
        self._last_error_code: str | None = None
        self._last_error_message: str | None = None
        self._session_ended_sent: bool = False
        self._persistence_finalized: bool = False
        self._turn_persist_lock = asyncio.Lock()
        self.initialized_ok: bool = False

    def _is_ws_connected(self) -> bool:
        return (
            self.websocket.client_state == WebSocketState.CONNECTED
            and self.websocket.application_state == WebSocketState.CONNECTED
        )

    async def _send_event(
        self,
        event_type: ServerEventType,
        payload_model: Any,
    ) -> None:
        if not self._is_ws_connected():
            return
        payload_dict = (
            payload_model.model_dump(mode="json")
            if hasattr(payload_model, "model_dump")
            else dict(payload_model)
        )
        envelope = ServerEventEnvelope(
            event=event_type,
            session_id=self.session.session_id,
            payload=payload_dict,
        )
        async with self._send_lock:
            if self._is_ws_connected():
                await self.websocket.send_json(envelope.model_dump(mode="json"))

    async def _send_binary_audio(self, pcm_chunk: bytes) -> None:
        if not self._is_ws_connected() or self._current_turn_interrupted:
            return
        async with self._send_lock:
            if self._is_ws_connected() and not self._current_turn_interrupted:
                await self.websocket.send_bytes(pcm_chunk)

    def _flush_outbound_audio_queue(self) -> None:
        while not self._outbound_audio_queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._outbound_audio_queue.get_nowait()
                self._outbound_audio_queue.task_done()

    async def _emit_session_error(
        self,
        code: str,
        message: str,
        *,
        fatal: bool = False,
    ) -> None:
        if fatal:
            self._last_error_code = code
            self._last_error_message = message
        with contextlib.suppress(Exception):
            await self._send_event(
                ServerEventType.SESSION_ERROR,
                SessionErrorPayload(code=code, message=message, fatal=fatal),
            )

    async def _emit_session_ended(
        self,
        reason: Literal[
            "client_requested",
            "client_disconnected",
            "max_duration_reached",
            "provider_closed",
            "error",
        ],
    ) -> None:
        self._finalize_persistence_sync()
        if self._session_ended_sent:
            return
        self._session_ended_sent = True
        elapsed = round(max(0.0, time.monotonic() - self._start_monotonic), 2)
        with contextlib.suppress(Exception):
            await self._send_event(
                ServerEventType.SESSION_ENDED,
                SessionEndedPayload(reason=reason, duration_seconds=elapsed),
            )

    async def run(self) -> bool:
        """
        Executes the full gateway lifecycle.

        Returns True if Gemini Live initialized successfully, or False if initialization
        failed prior to session start (allowing the caller to refund daily quota).
        """
        self._start_monotonic = time.monotonic()
        try:
            async with self.live_client_factory(self.session, self.settings) as gemini_session:
                self.initialized_ok = True
                await self._on_gemini_connected(gemini_session)
                return True
        except WebSocketDisconnect:
            self._end_reason = "client_disconnected"
            return self.initialized_ok
        except genai_errors.APIError as exc:
            logger.warning(
                "Gemini Live API error during voice session",
                extra={
                    "extra_data": {
                        "session_id": str(self.session.session_id),
                        "user_id": str(self.session.user_id),
                        "status_code": getattr(exc, "code", None),
                    }
                },
            )
            self._end_reason = "error"
            await self._emit_session_error(
                "GEMINI_LIVE_UNAVAILABLE",
                "Voice Practice AI service encountered a connection error. Please try again.",
                fatal=True,
            )
            await self._emit_session_ended("error")
            return self.initialized_ok
        except Exception as exc:
            logger.error(
                "Unexpected error initializing or running Gemini Live session",
                extra={
                    "extra_data": {
                        "session_id": str(self.session.session_id),
                        "user_id": str(self.session.user_id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            self._end_reason = "error"
            await self._emit_session_error(
                "VOICE_SESSION_INIT_FAILED",
                "Unable to establish a live voice session at this time.",
                fatal=True,
            )
            await self._emit_session_ended("error")
            return self.initialized_ok
        finally:
            self._finalize_persistence_sync()
            self._cleanup_memory_state()

    async def _on_gemini_connected(
        self,
        gemini_session: GeminiLiveSessionProtocol,
    ) -> None:
        await self._send_event(
            ServerEventType.SESSION_READY,
            SessionReadyPayload(
                mode=self.session.mode,
                model=self.settings.GEMINI_LIVE_MODEL,
                audio_format=VoiceAudioFormatSpec(
                    input_sample_rate_hz=self._input_sample_rate_hz,
                    output_sample_rate_hz=GEMINI_OUTPUT_SAMPLE_RATE_HZ,
                ),
                max_duration_seconds=self.settings.VOICE_MAX_SESSION_DURATION_SECONDS,
                warning_at_seconds=self.settings.VOICE_SESSION_WARNING_SECONDS,
                max_audio_frame_bytes=self.settings.VOICE_MAX_AUDIO_FRAME_BYTES,
            ),
        )

        client_task = asyncio.create_task(
            self._client_receive_loop(gemini_session),
            name=f"voice-client-recv-{self.session.session_id}",
        )
        gemini_task = asyncio.create_task(
            self._gemini_receive_loop(gemini_session),
            name=f"voice-gemini-recv-{self.session.session_id}",
        )
        audio_sender_task = asyncio.create_task(
            self._outbound_audio_loop(),
            name=f"voice-audio-out-{self.session.session_id}",
        )
        timer_task = asyncio.create_task(
            self._session_timer_loop(),
            name=f"voice-timer-{self.session.session_id}",
        )
        stop_wait_task = asyncio.create_task(
            self._stop_event.wait(),
            name=f"voice-stop-wait-{self.session.session_id}",
        )

        tasks = {client_task, gemini_task, audio_sender_task, timer_task, stop_wait_task}
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._stop_event.set()
            self._flush_outbound_audio_queue()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            if self._end_reason != "client_disconnected":
                await self._emit_session_ended(self._end_reason)

    async def _ensure_session_started_emitted(self) -> None:
        if not self._session_started_emitted:
            self._session_started_emitted = True
            await self._send_event(
                ServerEventType.SESSION_STARTED,
                SessionStartedPayload(
                    mode=self.session.mode,
                    input_sample_rate_hz=self._input_sample_rate_hz,
                    output_sample_rate_hz=GEMINI_OUTPUT_SAMPLE_RATE_HZ,
                ),
            )

    async def _client_receive_loop(
        self,
        gemini_session: GeminiLiveSessionProtocol,
    ) -> None:
        while not self._stop_event.is_set():
            try:
                message = await self.websocket.receive()
            except WebSocketDisconnect:
                self._end_reason = "client_disconnected"
                self._stop_event.set()
                return
            except RuntimeError:
                self._end_reason = "client_disconnected"
                self._stop_event.set()
                return

            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                self._end_reason = "client_disconnected"
                self._stop_event.set()
                return

            if "bytes" in message and message["bytes"] is not None:
                await self._handle_client_audio_bytes(
                    gemini_session,
                    message["bytes"],
                )
            elif "text" in message and message["text"] is not None:
                await self._handle_client_text_message(
                    gemini_session,
                    message["text"],
                )

    async def _handle_client_audio_bytes(
        self,
        gemini_session: GeminiLiveSessionProtocol,
        pcm_bytes: bytes,
    ) -> None:
        frame_len = len(pcm_bytes)
        if frame_len == 0:
            await self._emit_session_error(
                "EMPTY_AUDIO_FRAME",
                "Audio frame must not be empty.",
                fatal=False,
            )
            return

        if frame_len > self.settings.VOICE_MAX_AUDIO_FRAME_BYTES:
            max_bytes = self.settings.VOICE_MAX_AUDIO_FRAME_BYTES
            await self._emit_session_error(
                "AUDIO_FRAME_TOO_LARGE",
                f"Audio frame exceeds maximum size of {max_bytes} bytes.",
                fatal=False,
            )
            return

        if frame_len % PCM_BYTES_PER_SAMPLE != 0:
            await self._emit_session_error(
                "INVALID_AUDIO_FRAME",
                "PCM 16-bit audio frame must have an even byte length.",
                fatal=False,
            )
            return

        try:
            normalized_pcm = normalize_pcm16le_to_16khz(pcm_bytes, self._input_sample_rate_hz)
        except ValueError as exc:
            await self._emit_session_error(
                "UNSUPPORTED_SAMPLE_RATE",
                str(exc),
                fatal=False,
            )
            return

        await self._ensure_session_started_emitted()
        self._audio_stream_end_sent_for_turn = False

        try:
            await gemini_session.send_audio(
                normalized_pcm,
                sample_rate_hz=DEFAULT_INPUT_SAMPLE_RATE_HZ,
            )
        except Exception as exc:
            logger.warning(
                "Failed forwarding audio frame to Gemini Live",
                extra={
                    "extra_data": {
                        "session_id": str(self.session.session_id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            await self._emit_session_error(
                "UPSTREAM_AUDIO_SEND_FAILED",
                "Failed to stream audio to the AI service.",
                fatal=True,
            )
            self._end_reason = "error"
            self._stop_event.set()

    async def _handle_client_text_message(
        self,
        gemini_session: GeminiLiveSessionProtocol,
        raw_text: str,
    ) -> None:
        if len(raw_text.encode("utf-8")) > self.settings.VOICE_MAX_JSON_MESSAGE_BYTES:
            await self._emit_session_error(
                "CONTROL_MESSAGE_TOO_LARGE",
                "Control message exceeds maximum allowed size.",
                fatal=False,
            )
            return

        try:
            parsed_json = json.loads(raw_text)
            if not isinstance(parsed_json, dict):
                raise ValueError("Control message must be a JSON object")
            control = ClientControlMessage.model_validate(parsed_json)
        except (json.JSONDecodeError, ValueError, ValidationError):
            await self._emit_session_error(
                "INVALID_CONTROL_MESSAGE",
                "Malformed JSON control message.",
                fatal=False,
            )
            return

        event = control.event
        payload = control.payload

        if event in (ClientEventType.SESSION_START, ClientEventType.SESSION_STARTED):
            requested_rate = payload.get("input_sample_rate_hz")
            if requested_rate is not None:
                if (
                    isinstance(requested_rate, int)
                    and not isinstance(requested_rate, bool)
                    and requested_rate in SUPPORTED_INPUT_SAMPLE_RATES
                ):
                    self._input_sample_rate_hz = requested_rate
                else:
                    await self._emit_session_error(
                        "UNSUPPORTED_SAMPLE_RATE",
                        f"Supported input sample rates (Hz): {SUPPORTED_INPUT_SAMPLE_RATES}",
                        fatal=False,
                    )
                    return
            self._session_started_emitted = False
            await self._ensure_session_started_emitted()
            self._audio_stream_end_sent_for_turn = False

        elif event == ClientEventType.AUDIO_INPUT_STARTED:
            requested_rate = payload.get("sample_rate_hz")
            if requested_rate is not None:
                if (
                    isinstance(requested_rate, int)
                    and not isinstance(requested_rate, bool)
                    and requested_rate in SUPPORTED_INPUT_SAMPLE_RATES
                ):
                    self._input_sample_rate_hz = requested_rate
                else:
                    await self._emit_session_error(
                        "UNSUPPORTED_SAMPLE_RATE",
                        f"Supported input sample rates (Hz): {SUPPORTED_INPUT_SAMPLE_RATES}",
                        fatal=False,
                    )
                    return
            await self._ensure_session_started_emitted()
            self._audio_stream_end_sent_for_turn = False
            self._current_turn_interrupted = False
            await self._send_event(
                ServerEventType.AUDIO_INPUT_STARTED,
                AudioInputStartedPayload(sample_rate_hz=self._input_sample_rate_hz),
            )

        elif event in (ClientEventType.AUDIO_INPUT_STOPPED, "audio_input_end"):
            if (
                self._audio_stream_end_sent_for_turn
                or self._audio_output_active
                or self._current_turn_interrupted
            ):
                logger.debug(
                    "Ignored duplicate or inactive audio_stream_end signal",
                    extra={"extra_data": {"session_id": str(self.session.session_id)}},
                )
                return

            self._audio_stream_end_sent_for_turn = True
            try:
                # Under the official google-genai SDK, this delegates directly to
                # session.send_realtime_input(audio_stream_end=True)
                await gemini_session.send_audio_stream_end()
            except Exception as exc:
                logger.warning(
                    "Non-fatal error sending audio_stream_end to Gemini Live",
                    extra={
                        "extra_data": {
                            "session_id": str(self.session.session_id),
                            "error_type": type(exc).__name__,
                        }
                    },
                )

            await self._send_event(
                ServerEventType.AUDIO_INPUT_STOPPED,
                AudioInputStoppedPayload(turn_index=self._turn_index),
            )

        elif event == ClientEventType.INTERRUPT:
            await self._handle_interruption(source="client_signal")

        elif event == ClientEventType.PING:
            await self._send_event(
                ServerEventType.PONG,
                {"client_timestamp": payload.get("timestamp")},
            )

        elif event == ClientEventType.SESSION_END:
            self._end_reason = "client_requested"
            self._stop_event.set()

    async def _outbound_audio_loop(self) -> None:
        while not self._stop_event.is_set():
            chunk = await self._outbound_audio_queue.get()
            try:
                if not self._current_turn_interrupted:
                    await self._send_binary_audio(chunk)
            finally:
                self._outbound_audio_queue.task_done()

    async def _drain_outbound_audio(self) -> None:
        if not self._outbound_audio_queue.empty():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._outbound_audio_queue.join(), timeout=1.0)

    async def _handle_interruption(
        self,
        *,
        source: Literal["user_speech", "client_signal"],
    ) -> None:
        self._current_turn_interrupted = True
        self._had_any_interruption = True
        self._flush_outbound_audio_queue()

        # Persist any in-progress or finalized assistant transcript as interrupted
        await self._finalize_assistant_transcript_if_pending(
            turn_status="interrupted",
            force_status_update=True,
        )

        if self._audio_output_active:
            self._audio_output_active = False
            await self._send_event(
                ServerEventType.AUDIO_OUTPUT_FINISHED,
                AudioOutputFinishedPayload(
                    turn_index=self._turn_index,
                    reason="interrupted",
                ),
            )

        await self._send_event(
            ServerEventType.INTERRUPTED,
            InterruptedPayload(
                turn_index=self._turn_index,
                source=source,
            ),
        )

    def _allocate_persisted_turn_index(self) -> int:
        self._persisted_turn_counter += 1
        return self._persisted_turn_counter

    async def _finalize_user_transcript_if_pending(
        self,
        *,
        turn_status: Literal["completed", "interrupted"] = "completed",
    ) -> None:
        if self._user_transcript_buffer and not self._user_transcript_finalized:
            full_text = "".join(self._user_transcript_buffer).strip()
            self._user_transcript_finalized = True
            if full_text:
                if self._user_persisted_turn_idx is None:
                    self._user_persisted_turn_idx = self._allocate_persisted_turn_index()
                persisted_idx = self._user_persisted_turn_idx
                started_at = self._user_turn_started_at or datetime.now(UTC)
                completed_at = datetime.now(UTC)
                await self._send_event(
                    ServerEventType.TRANSCRIPT_FINAL,
                    TranscriptEventPayload(
                        role="user",
                        text=full_text,
                        turn_index=self._turn_index,
                        is_final=True,
                    ),
                )
                async with self._turn_persist_lock:
                    await asyncio.to_thread(
                        self.persistence.save_turn,
                        session_id=self.session.session_id,
                        turn_index=persisted_idx,
                        speaker_role="user",
                        transcript_text=full_text,
                        turn_status=turn_status,
                        started_at=started_at,
                        completed_at=completed_at,
                        metadata_json={"exchange_index": self._turn_index},
                    )
            # Reset user buffer so multi-segment / long utterances finalize every segment cleanly
            self._user_transcript_buffer.clear()
            self._user_transcript_finalized = False
            self._user_turn_started_at = None
            self._user_persisted_turn_idx = None

    async def _finalize_assistant_transcript_if_pending(
        self,
        *,
        turn_status: Literal["completed", "interrupted"] | None = None,
        force_status_update: bool = False,
    ) -> None:
        effective_status: Literal["completed", "interrupted"] = turn_status or (
            "interrupted" if self._current_turn_interrupted else "completed"
        )
        if self._assistant_transcript_buffer and (
            not self._assistant_transcript_finalized or force_status_update
        ):
            full_text = "".join(self._assistant_transcript_buffer).strip()
            was_already_finalized = self._assistant_transcript_finalized
            self._assistant_transcript_finalized = True
            if full_text:
                if self._assistant_persisted_turn_idx is None:
                    self._assistant_persisted_turn_idx = self._allocate_persisted_turn_index()
                persisted_idx = self._assistant_persisted_turn_idx
                started_at = self._assistant_turn_started_at or datetime.now(UTC)
                completed_at = datetime.now(UTC)
                if not was_already_finalized:
                    await self._send_event(
                        ServerEventType.TRANSCRIPT_FINAL,
                        TranscriptEventPayload(
                            role="assistant",
                            text=full_text,
                            turn_index=self._turn_index,
                            is_final=True,
                        ),
                    )
                async with self._turn_persist_lock:
                    await asyncio.to_thread(
                        self.persistence.save_turn,
                        session_id=self.session.session_id,
                        turn_index=persisted_idx,
                        speaker_role="assistant",
                        transcript_text=full_text,
                        turn_status=effective_status,
                        started_at=started_at,
                        completed_at=completed_at,
                        metadata_json={"exchange_index": self._turn_index},
                    )

    def _finalize_persistence_sync(self) -> None:
        """
        Synchronously flushes any unfinalized user or assistant transcript buffers and records
        the final session state in PostgreSQL.

        Executed before emitting `session_ended` (so post-session REST endpoints immediately
        observe the finalized session and all turns) and guarded to run at most once.
        """
        if self._persistence_finalized:
            return
        self._persistence_finalized = True

        had_unfinalized = bool(
            (self._user_transcript_buffer and not self._user_transcript_finalized)
            or (self._assistant_transcript_buffer and not self._assistant_transcript_finalized)
        )
        if had_unfinalized:
            self._had_any_interruption = True

        exit_turn_status: Literal["completed", "interrupted"] = (
            "interrupted"
            if (
                had_unfinalized
                or self._end_reason in ("client_disconnected", "error")
                or self._current_turn_interrupted
            )
            else "completed"
        )
        now = datetime.now(UTC)

        if self._user_transcript_buffer and not self._user_transcript_finalized:
            user_text = "".join(self._user_transcript_buffer).strip()
            self._user_transcript_finalized = True
            if user_text:
                if self._user_persisted_turn_idx is None:
                    self._user_persisted_turn_idx = self._allocate_persisted_turn_index()
                with contextlib.suppress(Exception):
                    self.persistence.save_turn(
                        session_id=self.session.session_id,
                        turn_index=self._user_persisted_turn_idx,
                        speaker_role="user",
                        transcript_text=user_text,
                        turn_status=exit_turn_status,
                        started_at=self._user_turn_started_at or now,
                        completed_at=now,
                        metadata_json={"exchange_index": self._turn_index},
                    )

        if self._assistant_transcript_buffer and not self._assistant_transcript_finalized:
            assistant_text = "".join(self._assistant_transcript_buffer).strip()
            self._assistant_transcript_finalized = True
            if assistant_text:
                if self._assistant_persisted_turn_idx is None:
                    self._assistant_persisted_turn_idx = self._allocate_persisted_turn_index()
                with contextlib.suppress(Exception):
                    self.persistence.save_turn(
                        session_id=self.session.session_id,
                        turn_index=self._assistant_persisted_turn_idx,
                        speaker_role="assistant",
                        transcript_text=assistant_text,
                        turn_status=exit_turn_status,
                        started_at=self._assistant_turn_started_at or now,
                        completed_at=now,
                        metadata_json={"exchange_index": self._turn_index},
                    )

        elapsed = round(max(0.0, time.monotonic() - self._start_monotonic), 2)
        with contextlib.suppress(Exception):
            self.persistence.finalize_session(
                session_id=self.session.session_id,
                termination_reason=self._end_reason,
                duration_seconds=elapsed,
                had_interruption=self._had_any_interruption,
                error_code=self._last_error_code,
                error_message=self._last_error_message,
            )

    def _reset_turn_buffers(self) -> None:
        self._user_transcript_buffer.clear()
        self._user_transcript_finalized = False
        self._user_turn_started_at = None
        self._user_persisted_turn_idx = None

        self._assistant_transcript_buffer.clear()
        self._assistant_transcript_finalized = False
        self._assistant_turn_started_at = None
        self._assistant_persisted_turn_idx = None

        self._current_turn_interrupted = False
        self._audio_stream_end_sent_for_turn = False

    async def _gemini_receive_loop(
        self,
        gemini_session: GeminiLiveSessionProtocol,
    ) -> None:
        try:
            while not self._stop_event.is_set():
                received_any = False
                completed_turn_in_cycle = False

                async for message in gemini_session.receive():
                    if self._stop_event.is_set():
                        return
                    received_any = True
                    turn_done = await self._process_gemini_message(message)
                    if turn_done:
                        completed_turn_in_cycle = True

                # In google-genai, `session.receive()` breaks at the end of each turn
                # (`turn_complete=True`). If the iterator exited without completing a turn
                # or yielded nothing, the upstream stream has closed.
                if not received_any or not completed_turn_in_cycle:
                    self._end_reason = "provider_closed"
                    self._stop_event.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Gemini Live receive stream terminated unexpectedly",
                extra={
                    "extra_data": {
                        "session_id": str(self.session.session_id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            await self._emit_session_error(
                "GEMINI_STREAM_CLOSED",
                "The live audio connection to the AI service ended unexpectedly.",
                fatal=True,
            )
            self._end_reason = "error"
            self._stop_event.set()

    async def _process_gemini_message(self, message: Any) -> bool:
        go_away = getattr(message, "go_away", None)
        if go_away is not None:
            await self._send_event(
                ServerEventType.SESSION_WARNING,
                SessionWarningPayload(
                    remaining_seconds=5,
                    max_duration_seconds=self.settings.VOICE_MAX_SESSION_DURATION_SECONDS,
                    message="AI service connection will refresh shortly.",
                ),
            )

        server_content = getattr(message, "server_content", None)
        if server_content is None:
            return False

        await self._ensure_session_started_emitted()
        self._audio_stream_end_sent_for_turn = False

        # 1. Input (User) transcription events
        input_tx = getattr(server_content, "input_transcription", None)
        if input_tx is not None:
            tx_text = getattr(input_tx, "text", None)
            tx_finished = bool(getattr(input_tx, "finished", False))
            if isinstance(tx_text, str) and tx_text:
                # If a previous turn was interrupted, user speech starts a fresh turn context
                if self._current_turn_interrupted:
                    self._current_turn_interrupted = False
                self._audio_stream_end_sent_for_turn = False
                if self._user_transcript_finalized:
                    self._user_transcript_buffer.clear()
                    self._user_transcript_finalized = False
                    self._user_persisted_turn_idx = None
                if not self._user_transcript_buffer:
                    self._user_turn_started_at = datetime.now(UTC)
                self._user_transcript_buffer.append(tx_text)
                accumulated_user = "".join(self._user_transcript_buffer).strip()
                if accumulated_user:
                    await self._send_event(
                        ServerEventType.TRANSCRIPT_PARTIAL,
                        TranscriptEventPayload(
                            role="user",
                            text=accumulated_user,
                            turn_index=self._turn_index,
                            is_final=False,
                        ),
                    )
            if tx_finished:
                await self._finalize_user_transcript_if_pending()

        # 2. Server-side VAD interruption signal
        if getattr(server_content, "interrupted", False):
            await self._handle_interruption(source="user_speech")

        # 3. Model audio turn parts
        model_turn = getattr(server_content, "model_turn", None)
        parts = getattr(model_turn, "parts", None) if model_turn is not None else None
        if parts and not self._current_turn_interrupted:
            await self._finalize_user_transcript_if_pending()
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                audio_bytes = (
                    getattr(inline_data, "data", None) if inline_data is not None else None
                )
                if isinstance(audio_bytes, bytes) and len(audio_bytes) > 0:
                    if not self._audio_output_active:
                        self._audio_output_active = True
                        await self._send_event(
                            ServerEventType.AUDIO_OUTPUT_STARTED,
                            AudioOutputStartedPayload(
                                turn_index=self._turn_index,
                                sample_rate_hz=GEMINI_OUTPUT_SAMPLE_RATE_HZ,
                            ),
                        )
                    if not self._current_turn_interrupted:
                        await self._outbound_audio_queue.put(audio_bytes)

        # 4. Output (Assistant) transcription events
        output_tx = getattr(server_content, "output_transcription", None)
        if output_tx is not None and not self._current_turn_interrupted:
            out_text = getattr(output_tx, "text", None)
            out_finished = bool(getattr(output_tx, "finished", False))
            if isinstance(out_text, str) and out_text:
                if not self._assistant_transcript_buffer:
                    self._assistant_turn_started_at = datetime.now(UTC)
                self._assistant_transcript_buffer.append(out_text)
                accumulated_assistant = "".join(self._assistant_transcript_buffer).strip()
                if accumulated_assistant:
                    await self._send_event(
                        ServerEventType.TRANSCRIPT_PARTIAL,
                        TranscriptEventPayload(
                            role="assistant",
                            text=accumulated_assistant,
                            turn_index=self._turn_index,
                            is_final=False,
                        ),
                    )
            if out_finished:
                await self._finalize_assistant_transcript_if_pending()

        # 5. Turn completion
        if getattr(server_content, "turn_complete", False):
            if not self._current_turn_interrupted:
                await self._drain_outbound_audio()
            await self._finalize_user_transcript_if_pending()
            await self._finalize_assistant_transcript_if_pending()

            if self._audio_output_active:
                self._audio_output_active = False
                await self._send_event(
                    ServerEventType.AUDIO_OUTPUT_FINISHED,
                    AudioOutputFinishedPayload(
                        turn_index=self._turn_index,
                        reason="interrupted" if self._current_turn_interrupted else "completed",
                    ),
                )

            was_interrupted = self._current_turn_interrupted
            completed_turn_idx = self._turn_index
            self._turn_index += 1
            self._reset_turn_buffers()

            await self._send_event(
                ServerEventType.TURN_COMPLETED,
                TurnCompletedPayload(
                    turn_index=completed_turn_idx,
                    interrupted=was_interrupted,
                ),
            )
            return True

        return False

    async def _session_timer_loop(self) -> None:
        max_duration = float(self.settings.VOICE_MAX_SESSION_DURATION_SECONDS)
        warning_at = float(self.settings.VOICE_SESSION_WARNING_SECONDS)

        if 0 < warning_at < max_duration:
            await asyncio.sleep(warning_at)
            if self._stop_event.is_set():
                return
            remaining = max(1, int(round(max_duration - warning_at)))
            await self._send_event(
                ServerEventType.SESSION_WARNING,
                SessionWarningPayload(
                    remaining_seconds=remaining,
                    max_duration_seconds=int(max_duration),
                    message=f"Voice practice session will end in {remaining} seconds.",
                ),
            )
            await asyncio.sleep(max(0.0, max_duration - warning_at))
        else:
            await asyncio.sleep(max_duration)

        if not self._stop_event.is_set():
            logger.info(
                "Voice Practice session reached authoritative maximum duration",
                extra={
                    "extra_data": {
                        "session_id": str(self.session.session_id),
                        "user_id": str(self.session.user_id),
                        "max_duration_seconds": int(max_duration),
                    }
                },
            )
            self._end_reason = "max_duration_reached"
            self._stop_event.set()

    def _cleanup_memory_state(self) -> None:
        self._flush_outbound_audio_queue()
        self._user_transcript_buffer.clear()
        self._assistant_transcript_buffer.clear()
        self._audio_output_active = False
