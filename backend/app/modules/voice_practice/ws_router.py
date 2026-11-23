import contextlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Query, WebSocket, status
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketState

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.voice_practice.evaluator import (
    VoicePracticeEvaluator,
    create_voice_practice_evaluator,
)
from app.modules.voice_practice.gateway import (
    GeminiLiveClientFactory,
    VoicePracticeGateway,
    get_gemini_live_client_factory,
)
from app.modules.voice_practice.schemas import (
    ServerEventEnvelope,
    ServerEventType,
    SessionErrorPayload,
    VoicePracticeFeedbackRead,
    VoicePracticeMode,
    VoicePracticeSessionDeleteResponse,
    VoicePracticeSessionDetailRead,
    VoicePracticeSessionListResponse,
    VoicePracticeTurnRead,
    VoiceSessionTicketRequest,
    VoiceSessionTicketResponse,
)
from app.modules.voice_practice.service import (
    VoicePracticeService,
    VoiceSessionStore,
    get_voice_session_store,
)
from app.shared.responses import ApiResponse

logger = logging.getLogger("app.modules.voice_practice.ws_router")

voice_practice_router = APIRouter(
    prefix="/voice-practice",
    tags=["Voice Practice"],
)

_FORBIDDEN_QUERY_PARAMS = {"token", "access_token", "jwt", "api_key", "gemini_api_key", "key"}


def get_voice_practice_evaluator(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VoicePracticeEvaluator:
    return create_voice_practice_evaluator(settings)


def get_voice_practice_service(
    db: Annotated[Session, Depends(get_db)],
    store: Annotated[VoiceSessionStore, Depends(get_voice_session_store)],
    evaluator: Annotated[VoicePracticeEvaluator, Depends(get_voice_practice_evaluator)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VoicePracticeService:
    return VoicePracticeService(
        db,
        store=store,
        evaluator=evaluator,
        settings=settings,
    )


VoicePracticeServiceDep = Annotated[VoicePracticeService, Depends(get_voice_practice_service)]


def _looks_like_jwt(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 3 and value.startswith("eyJ")


async def _reject_websocket(
    websocket: WebSocket,
    *,
    code: str,
    message: str,
    close_code: int = status.WS_1008_POLICY_VIOLATION,
    session_id: uuid.UUID | None = None,
) -> None:
    with contextlib.suppress(Exception):
        if websocket.application_state == WebSocketState.CONNECTING:
            await websocket.accept()
        if websocket.application_state == WebSocketState.CONNECTED:
            envelope = ServerEventEnvelope(
                event=ServerEventType.SESSION_ERROR,
                session_id=session_id or uuid.UUID(int=0),
                timestamp=datetime.now(UTC),
                payload=SessionErrorPayload(
                    code=code,
                    message=message,
                    fatal=True,
                ).model_dump(mode="json"),
            )
            await websocket.send_json(envelope.model_dump(mode="json"))
            await websocket.close(code=close_code, reason=code[:120])


@voice_practice_router.post(
    "/session-ticket",
    response_model=ApiResponse[VoiceSessionTicketResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_voice_session_ticket(
    current_user: CurrentUser,
    service: VoicePracticeServiceDep,
    payload: Annotated[VoiceSessionTicketRequest | None, Body()] = None,
) -> ApiResponse[VoiceSessionTicketResponse]:
    """
    Issues a short-lived (60s), single-use connection ticket for the Voice Practice WebSocket.

    Validates user account status, Gemini configuration, concurrency limits, and daily session
    quota before issuing the ticket.
    """
    effective_payload = payload or VoiceSessionTicketRequest()
    ticket_data = service.issue_connection_ticket(current_user, effective_payload)
    return ApiResponse(
        data=ticket_data,
        message="Voice practice connection ticket issued",
    )


@voice_practice_router.websocket("/ws")
async def voice_practice_websocket(
    websocket: WebSocket,
    service: VoicePracticeServiceDep,
    settings: Annotated[Settings, Depends(get_settings)],
    live_client_factory: Annotated[
        GeminiLiveClientFactory,
        Depends(get_gemini_live_client_factory),
    ],
    ticket: Annotated[str | None, Query()] = None,
) -> None:
    """
    Bidirectional WebSocket gateway bridging mobile client PCM audio and Gemini Live.
    """
    # 1. Reject forbidden query parameters (long-lived JWTs or API keys in URL)
    query_keys = {k.lower() for k in websocket.query_params}
    if query_keys & _FORBIDDEN_QUERY_PARAMS:
        logger.warning("Rejected voice WebSocket connection with forbidden query parameter")
        await _reject_websocket(
            websocket,
            code="FORBIDDEN_QUERY_PARAMETER",
            message=(
                "Long-lived tokens and API keys are not permitted in WebSocket URLs. "
                "Use a single-use connection ticket."
            ),
        )
        return

    header_ticket = websocket.headers.get("X-Voice-Ticket")
    raw_ticket = (ticket or header_ticket or "").strip()

    if not raw_ticket:
        await _reject_websocket(
            websocket,
            code="MISSING_CONNECTION_TICKET",
            message="A valid single-use connection ticket is required.",
        )
        return

    if _looks_like_jwt(raw_ticket):
        logger.warning("Rejected voice WebSocket connection attempting to use JWT as ticket")
        await _reject_websocket(
            websocket,
            code="JWT_NOT_ALLOWED_IN_WS",
            message=(
                "Do not pass JWT access tokens to the WebSocket. "
                "Exchange for a session ticket first."
            ),
        )
        return

    # 2. Redeem single-use ticket and acquire concurrency lock BEFORE allocating Gemini resources
    try:
        active_session = service.redeem_ticket_and_acquire_session(raw_ticket)
    except AppException as exc:
        await _reject_websocket(
            websocket,
            code=exc.code,
            message=exc.message,
        )
        return
    except Exception as exc:
        logger.error(
            "Unexpected error during WebSocket ticket redemption",
            extra={"extra_data": {"error_type": type(exc).__name__}},
        )
        await _reject_websocket(
            websocket,
            code="INTERNAL_AUTH_ERROR",
            message="Unable to authorize voice session.",
            close_code=status.WS_1011_INTERNAL_ERROR,
        )
        return
    finally:
        # Release SQLAlchemy DB connection back to pool before long-lived WebSocket streaming
        with contextlib.suppress(Exception):
            service.db.close()

    # 3. Accept WebSocket and run Gemini Live bridge with guaranteed resource cleanup
    initialized_ok = False
    gateway: VoicePracticeGateway | None = None
    try:
        await websocket.accept()
        gateway = VoicePracticeGateway(
            websocket=websocket,
            session=active_session,
            settings=settings,
            live_client_factory=live_client_factory,
        )
        initialized_ok = await gateway.run()
    except Exception as exc:
        logger.error(
            "Unhandled exception in VoicePracticeGateway",
            extra={
                "extra_data": {
                    "session_id": str(active_session.session_id),
                    "user_id": str(active_session.user_id),
                    "error_type": type(exc).__name__,
                }
            },
        )
    finally:
        if gateway is not None and gateway.initialized_ok:
            initialized_ok = True
        service.release_session(
            active_session.session_id,
            active_session.user_id,
            refund_daily_quota=not initialized_ok,
        )
        with contextlib.suppress(Exception):
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)


# ============================================================================
# VOICE PRACTICE SESSION HISTORY, TURNS, FEEDBACK & EVALUATION ENDPOINTS
# ============================================================================


@voice_practice_router.get(
    "/sessions",
    response_model=ApiResponse[VoicePracticeSessionListResponse],
)
def list_voice_practice_sessions(
    current_user: CurrentUser,
    service: VoicePracticeServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    mode: Annotated[VoicePracticeMode | None, Query()] = None,
    session_status: Annotated[str | None, Query(alias="status")] = None,
) -> ApiResponse[VoicePracticeSessionListResponse]:
    """Lists the authenticated user's Voice Practice sessions ordered by newest first."""
    result = service.list_sessions(
        current_user.id,
        limit=limit,
        offset=offset,
        mode=mode.value if mode is not None else None,
        session_status=session_status,
    )
    return ApiResponse(
        data=result,
        message="Voice practice sessions retrieved",
    )


@voice_practice_router.get(
    "/sessions/{session_id}",
    response_model=ApiResponse[VoicePracticeSessionDetailRead],
)
def get_voice_practice_session(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    service: VoicePracticeServiceDep,
) -> ApiResponse[VoicePracticeSessionDetailRead]:
    """Retrieves full details, configuration, turns, and feedback for a Voice Practice session."""
    result = service.get_session(current_user.id, session_id)
    return ApiResponse(
        data=result,
        message="Voice practice session retrieved",
    )


@voice_practice_router.get(
    "/sessions/{session_id}/turns",
    response_model=ApiResponse[list[VoicePracticeTurnRead]],
)
def get_voice_practice_session_turns(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    service: VoicePracticeServiceDep,
) -> ApiResponse[list[VoicePracticeTurnRead]]:
    """Retrieves ordered finalized transcript turns for a Voice Practice session."""
    turns = service.get_session_turns(current_user.id, session_id)
    return ApiResponse(
        data=turns,
        message="Voice practice session turns retrieved",
    )


@voice_practice_router.get(
    "/sessions/{session_id}/feedback",
    response_model=ApiResponse[VoicePracticeFeedbackRead],
)
def get_voice_practice_session_feedback(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    service: VoicePracticeServiceDep,
) -> ApiResponse[VoicePracticeFeedbackRead]:
    """Retrieves persisted structured evaluation feedback for a Voice Practice session."""
    feedback = service.get_session_feedback(current_user.id, session_id)
    return ApiResponse(
        data=feedback,
        message="Voice practice session feedback retrieved",
    )


@voice_practice_router.post(
    "/sessions/{session_id}/evaluate",
    response_model=ApiResponse[VoicePracticeFeedbackRead],
)
def evaluate_voice_practice_session(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    service: VoicePracticeServiceDep,
) -> ApiResponse[VoicePracticeFeedbackRead]:
    """
    Triggers or safely retries post-session structured evaluation for a completed or interrupted
    Voice Practice session.
    """
    feedback = service.evaluate_session(current_user.id, session_id)
    return ApiResponse(
        data=feedback,
        message="Voice practice session evaluated",
    )


@voice_practice_router.delete(
    "/sessions/{session_id}",
    response_model=ApiResponse[VoicePracticeSessionDeleteResponse],
)
def delete_voice_practice_session(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    service: VoicePracticeServiceDep,
) -> ApiResponse[VoicePracticeSessionDeleteResponse]:
    """Deletes a Voice Practice session and cascades deletion to turns, feedback, and links."""
    result = service.delete_session(current_user.id, session_id)
    return ApiResponse(
        data=result,
        message="Voice practice session deleted",
    )
