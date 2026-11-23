import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.modules.documents.models import Document


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatConversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Represents an AI document chat conversation scoped to 1..N user-owned learning documents.
    """

    __tablename__ = "chat_conversations"
    __table_args__ = (
        Index("ix_chat_conversations_user_id", "user_id"),
        Index("ix_chat_conversations_updated_at", "updated_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Conversation",
    )

    conversation_documents: Mapped[list["ChatConversationDocument"]] = relationship(
        "ChatConversationDocument",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatConversationDocument.position.asc()",
    )
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="[ChatMessage.created_at.asc(), ChatMessage.id.asc()]",
    )


class ChatConversationDocument(Base, UUIDPrimaryKeyMixin):
    """
    Associates a chat conversation with a user learning document.
    """

    __tablename__ = "chat_conversation_documents"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "document_id",
            name="uq_chat_conversation_documents_conv_doc",
        ),
        CheckConstraint("position >= 0", name="chat_doc_position_non_negative"),
        Index("ix_chat_conversation_documents_conversation_id", "conversation_id"),
        Index("ix_chat_conversation_documents_document_id", "document_id"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    conversation: Mapped["ChatConversation"] = relationship(
        "ChatConversation",
        back_populates="conversation_documents",
    )
    document: Mapped["Document"] = relationship("Document")


class ChatMessage(Base, UUIDPrimaryKeyMixin):
    """
    A single message turn in a chat conversation (either 'user' or 'assistant').
    """

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="check_chat_message_role",
        ),
        Index("ix_chat_messages_conversation_id", "conversation_id"),
        Index("ix_chat_messages_created_at", "created_at"),
        Index("ix_chat_messages_conv_created", "conversation_id", "created_at", "id"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    conversation: Mapped["ChatConversation"] = relationship(
        "ChatConversation",
        back_populates="messages",
    )

