import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.chat.models import (
    ChatConversation,
    ChatConversationDocument,
    ChatMessage,
)


class ChatRepository:
    """
    Data access repository for chat conversations, associated documents, and messages.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_conversation(
        self,
        *,
        user_id: uuid.UUID,
        title: str,
        document_ids: Sequence[uuid.UUID],
    ) -> ChatConversation:
        conversation = ChatConversation(
            user_id=user_id,
            title=title,
        )
        self.db.add(conversation)
        self.db.flush()

        for pos, doc_id in enumerate(document_ids):
            link = ChatConversationDocument(
                conversation_id=conversation.id,
                document_id=doc_id,
                position=pos,
            )
            self.db.add(link)

        self.db.flush()
        return conversation

    def get_conversation_for_user(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> ChatConversation | None:
        stmt = (
            select(ChatConversation)
            .options(
                selectinload(ChatConversation.conversation_documents).selectinload(
                    ChatConversationDocument.document
                ),
                selectinload(ChatConversation.messages),
            )
            .where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        )
        return self.db.scalar(stmt)

    def list_conversations_for_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ChatConversation], int]:
        count_stmt = (
            select(func.count())
            .select_from(ChatConversation)
            .where(ChatConversation.user_id == user_id)
        )
        total = self.db.scalar(count_stmt) or 0

        stmt = (
            select(ChatConversation)
            .options(
                selectinload(ChatConversation.conversation_documents).selectinload(
                    ChatConversationDocument.document
                ),
                selectinload(ChatConversation.messages),
            )
            .where(ChatConversation.user_id == user_id)
            .order_by(ChatConversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list(self.db.scalars(stmt).all())
        return items, total

    def save_message(
        self,
        *,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        created_at: datetime | None = None,
    ) -> ChatMessage:
        kwargs: dict[str, Any] = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "citations": citations,
        }
        if created_at is not None:
            kwargs["created_at"] = created_at
        msg = ChatMessage(**kwargs)
        self.db.add(msg)
        self.db.flush()
        return msg

    def get_recent_messages(
        self,
        conversation_id: uuid.UUID,
        *,
        limit: int = 6,
        exclude_message_id: uuid.UUID | None = None,
    ) -> list[ChatMessage]:
        stmt = select(ChatMessage).where(ChatMessage.conversation_id == conversation_id)
        if exclude_message_id is not None:
            stmt = stmt.where(ChatMessage.id != exclude_message_id)

        stmt = stmt.order_by(
            ChatMessage.created_at.desc(),
            ChatMessage.id.desc(),
        ).limit(limit)
        messages = list(self.db.scalars(stmt).all())
        # Return in chronological order
        messages.reverse()
        return messages

    def update_conversation(
        self,
        conversation: ChatConversation,
        *,
        title: str | None = None,
    ) -> None:
        if title:
            conversation.title = title
        conversation.updated_at = datetime.now(UTC)
        self.db.flush()

    def delete_conversation(
        self,
        conversation: ChatConversation,
    ) -> None:
        self.db.delete(conversation)
        self.db.flush()
