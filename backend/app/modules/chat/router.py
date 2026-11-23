import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.chat.schemas import (
    ChatConversationCreateRequest,
    ChatConversationDeleteResponse,
    ChatConversationDetailRead,
    ChatConversationListResponse,
    ChatMessageRead,
    ChatMessageSendRequest,
    ChatMessageSendResponse,
)
from app.modules.chat.service import ChatService
from app.shared.responses import ApiResponse

chat_router = APIRouter(prefix="/chat", tags=["Chat"])


def get_chat_service(
    db: Annotated[Session, Depends(get_db)],
) -> ChatService:
    return ChatService(db)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@chat_router.post(
    "/conversations",
    response_model=ApiResponse[ChatConversationDetailRead],
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    payload: ChatConversationCreateRequest,
    current_user: CurrentUser,
    service: ChatServiceDep,
) -> ApiResponse[ChatConversationDetailRead]:
    conversation = service.create_conversation(current_user.id, payload)
    return ApiResponse(
        data=conversation,
        message="Conversation created successfully",
    )


@chat_router.get(
    "/conversations",
    response_model=ApiResponse[ChatConversationListResponse],
    status_code=status.HTTP_200_OK,
)
def list_conversations(
    current_user: CurrentUser,
    service: ChatServiceDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ChatConversationListResponse]:
    result = service.list_conversations(
        current_user.id,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(data=result)


@chat_router.get(
    "/conversations/{conversation_id}",
    response_model=ApiResponse[ChatConversationDetailRead],
    status_code=status.HTTP_200_OK,
)
def get_conversation(
    conversation_id: uuid.UUID,
    current_user: CurrentUser,
    service: ChatServiceDep,
) -> ApiResponse[ChatConversationDetailRead]:
    conversation = service.get_conversation(current_user.id, conversation_id)
    return ApiResponse(data=conversation)


@chat_router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ApiResponse[list[ChatMessageRead]],
    status_code=status.HTTP_200_OK,
)
def get_conversation_messages(
    conversation_id: uuid.UUID,
    current_user: CurrentUser,
    service: ChatServiceDep,
) -> ApiResponse[list[ChatMessageRead]]:
    conversation = service.get_conversation(current_user.id, conversation_id)
    return ApiResponse(data=conversation.messages)


@chat_router.post(
    "/conversations/{conversation_id}/messages",
    response_model=ApiResponse[ChatMessageSendResponse],
    status_code=status.HTTP_201_CREATED,
)
def send_message(
    conversation_id: uuid.UUID,
    payload: ChatMessageSendRequest,
    current_user: CurrentUser,
    service: ChatServiceDep,
) -> ApiResponse[ChatMessageSendResponse]:
    result = service.send_message(
        user_id=current_user.id,
        conversation_id=conversation_id,
        message_text=payload.message,
    )
    return ApiResponse(
        data=result,
        message="Message processed successfully",
    )


@chat_router.delete(
    "/conversations/{conversation_id}",
    response_model=ApiResponse[ChatConversationDeleteResponse],
    status_code=status.HTTP_200_OK,
)
def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: CurrentUser,
    service: ChatServiceDep,
) -> ApiResponse[ChatConversationDeleteResponse]:
    service.delete_conversation(current_user.id, conversation_id)
    return ApiResponse(
        data=ChatConversationDeleteResponse(
            deleted=True,
            conversation_id=conversation_id,
        ),
        message="Conversation deleted successfully",
    )

