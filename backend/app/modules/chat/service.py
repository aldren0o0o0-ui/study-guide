import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    BadRequestException,
    NotFoundException,
    ServiceUnavailableException,
)
from app.modules.chat.exceptions import ChatError, ChatTransientError
from app.modules.chat.generator import (
    ChatGenerator,
    create_chat_generator,
)
from app.modules.chat.models import (
    ChatConversation,
    ChatRole,
)
from app.modules.chat.repository import ChatRepository
from app.modules.chat.schemas import (
    ChatConversationCreateRequest,
    ChatConversationDetailRead,
    ChatConversationDocumentRead,
    ChatConversationListResponse,
    ChatConversationSummaryRead,
    ChatMessageRead,
    ChatMessageSendResponse,
    ChatSourceCitation,
)
from app.modules.documents.models import Document, ProcessingStatus
from app.modules.processing.retrieval import RetrievalService

logger = logging.getLogger("app.modules.chat.service")


def _build_initial_title(
    documents: Sequence[Document],
    *,
    custom_title: str | None = None,
    initial_message: str | None = None,
) -> str:
    if custom_title:
        return custom_title[:255]
    if initial_message:
        snippet = " ".join(initial_message.strip().split())
        if len(snippet) <= 60:
            return snippet
        return snippet[:57].rstrip() + "..."
    if len(documents) == 1:
        return f"Chat: {documents[0].original_filename}"[:255]
    return f"Chat: {documents[0].original_filename} + {len(documents) - 1} more"[:255]


def _to_source_docs_read(conversation: ChatConversation) -> list[ChatConversationDocumentRead]:
    items: list[ChatConversationDocumentRead] = []
    for link in conversation.conversation_documents:
        if link.document is not None:
            items.append(
                ChatConversationDocumentRead(
                    id=link.id,
                    document_id=link.document_id,
                    original_filename=link.document.original_filename,
                    file_extension=link.document.file_extension,
                    processing_status=link.document.processing_status,
                )
            )
    return items


def _to_message_read(msg_model: Any) -> ChatMessageRead:
    citations_data: list[ChatSourceCitation] | None = None
    if msg_model.citations:
        citations_data = [
            ChatSourceCitation.model_validate(c) for c in msg_model.citations
        ]
    return ChatMessageRead(
        id=msg_model.id,
        conversation_id=msg_model.conversation_id,
        role=msg_model.role,
        content=msg_model.content,
        citations=citations_data,
        created_at=msg_model.created_at,
    )


def _to_summary_read(conversation: ChatConversation) -> ChatConversationSummaryRead:
    messages = conversation.messages or []
    latest_preview = messages[-1].content[:120] if messages else None
    return ChatConversationSummaryRead(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        documents_count=len(conversation.conversation_documents),
        messages_count=len(messages),
        latest_message_preview=latest_preview,
        source_documents=_to_source_docs_read(conversation),
    )


def _to_detail_read(conversation: ChatConversation) -> ChatConversationDetailRead:
    return ChatConversationDetailRead(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        source_documents=_to_source_docs_read(conversation),
        messages=[_to_message_read(m) for m in conversation.messages],
    )


class ChatService:
    def __init__(
        self,
        db: Session,
        *,
        retrieval_service: RetrievalService | None = None,
        generator: ChatGenerator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.repo = ChatRepository(db)
        self.retrieval_service = retrieval_service or RetrievalService(db, settings=self.settings)
        self.generator = generator or create_chat_generator(self.settings)

    def create_conversation(
        self,
        user_id: uuid.UUID,
        payload: ChatConversationCreateRequest,
    ) -> ChatConversationDetailRead:
        # Validate that all requested documents exist, belong to user, and are READY
        stmt = (
            select(Document)
            .where(
                Document.id.in_(payload.document_ids),
                Document.user_id == user_id,
            )
        )
        found_docs = list(self.db.scalars(stmt).all())
        found_map = {doc.id: doc for doc in found_docs}

        missing_ids = [d_id for d_id in payload.document_ids if d_id not in found_map]
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
                f"Documents must be fully processed before chat. Not ready: {unready_str}",
                code="DOCUMENTS_NOT_READY",
            )

        # Preserve selection order
        ordered_docs = [found_map[d_id] for d_id in payload.document_ids]
        title = _build_initial_title(
            ordered_docs,
            custom_title=payload.title,
            initial_message=payload.initial_message,
        )

        conversation = self.repo.create_conversation(
            user_id=user_id,
            title=title,
            document_ids=[doc.id for doc in ordered_docs],
        )

        try:
            # If initial message provided, process and respond immediately
            if payload.initial_message:
                self._execute_message_turn(
                    conversation=conversation,
                    user_id=user_id,
                    message_text=payload.initial_message,
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        # Refresh to populate relationships
        refreshed = self.repo.get_conversation_for_user(conversation.id, user_id)
        assert refreshed is not None
        return _to_detail_read(refreshed)

    def list_conversations(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> ChatConversationListResponse:
        items, total = self.repo.list_conversations_for_user(
            user_id,
            limit=limit,
            offset=offset,
        )
        return ChatConversationListResponse(
            items=[_to_summary_read(c) for c in items],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_conversation(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> ChatConversationDetailRead:
        conversation = self.repo.get_conversation_for_user(conversation_id, user_id)
        if conversation is None:
            raise NotFoundException("Chat conversation not found", code="CONVERSATION_NOT_FOUND")
        return _to_detail_read(conversation)

    def send_message(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        message_text: str,
    ) -> ChatMessageSendResponse:
        conversation = self.repo.get_conversation_for_user(conversation_id, user_id)
        if conversation is None:
            raise NotFoundException("Chat conversation not found", code="CONVERSATION_NOT_FOUND")

        try:
            user_msg, assistant_msg = self._execute_message_turn(
                conversation=conversation,
                user_id=user_id,
                message_text=message_text,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        return ChatMessageSendResponse(
            user_message=_to_message_read(user_msg),
            assistant_message=_to_message_read(assistant_msg),
            conversation_title=conversation.title,
        )

    def _execute_message_turn(
        self,
        *,
        conversation: ChatConversation,
        user_id: uuid.UUID,
        message_text: str,
    ) -> tuple[Any, Any]:
        # Stage 1: Read & Context Preparation (Read-Only)
        doc_ids = [
            link.document_id
            for link in conversation.conversation_documents
            if link.document is not None
            and link.document.processing_status == ProcessingStatus.READY.value
        ]

        if not doc_ids:
            # Fallback when all documents were removed/unready (no LLM generation needed)
            fallback_text = (
                "The documents associated with this conversation are no longer available or ready. "
                "Please start a new chat with active documents."
            )
            user_msg = self.repo.save_message(
                conversation_id=conversation.id,
                role=ChatRole.USER.value,
                content=message_text,
            )
            assistant_msg = self.repo.save_message(
                conversation_id=conversation.id,
                role=ChatRole.ASSISTANT.value,
                content=fallback_text,
                citations=None,
            )
            self.repo.update_conversation(conversation)
            return user_msg, assistant_msg

        # Retrieve relevant chunks via Phase 4 RetrievalService
        try:
            chunks = self.retrieval_service.retrieve_relevant_chunks(
                user_id=user_id,
                query=message_text,
                document_ids=doc_ids,
                limit=6,
            )
        except Exception as exc:
            logger.warning(
                "Retrieval failed during chat turn; continuing with empty chunks",
                extra={"extra_data": {"error": str(exc)}},
            )
            chunks = []

        # Fetch the true 6 prior messages in chronological order (user_msg is not yet saved)
        recent_msgs = self.repo.get_recent_messages(conversation.id, limit=6)
        history = [
            {"role": m.role, "content": m.content}
            for m in recent_msgs
        ]

        # Stage 2: AI Generation (External Network I/O outside write transaction)
        try:
            generated = self.generator.generate_answer(
                chunks=chunks,
                conversation_history=history,
                user_query=message_text,
            )
        except ChatTransientError as exc:
            raise ServiceUnavailableException(
                exc.user_message,
                code=exc.code,
            ) from exc
        except ChatError as exc:
            raise BadRequestException(
                exc.user_message,
                code=exc.code,
            ) from exc
        except Exception as exc:
            logger.error(
                "Unexpected failure during chat answer generation",
                extra={"extra_data": {"error": str(exc)}},
                exc_info=True,
            )
            raise ServiceUnavailableException(
                "The AI service is temporarily unavailable. Please try asking again.",
                code="CHAT_SERVICE_UNAVAILABLE",
            ) from exc

        # Stage 3: Atomic Persistence (Fast write transaction)
        turn_timestamp = datetime.now(UTC)
        user_msg = self.repo.save_message(
            conversation_id=conversation.id,
            role=ChatRole.USER.value,
            content=message_text,
            created_at=turn_timestamp,
        )

        citations_dump = (
            [c.model_dump(mode="json") for c in generated.citations]
            if generated.citations
            else None
        )
        assistant_msg = self.repo.save_message(
            conversation_id=conversation.id,
            role=ChatRole.ASSISTANT.value,
            content=generated.answer,
            citations=citations_dump,
            created_at=turn_timestamp + timedelta(milliseconds=1),
        )

        # Update conversation title on first question if using default title and bump updated_at
        new_title = None
        has_default_title = (
            conversation.title.startswith("New Conversation")
            or conversation.title.startswith("Chat: ")
        )
        if has_default_title:
            snippet = " ".join(message_text.strip().split())
            new_title = snippet[:57].rstrip() + "..." if len(snippet) > 60 else snippet

        self.repo.update_conversation(conversation, title=new_title)
        return user_msg, assistant_msg

    def delete_conversation(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> bool:
        conversation = self.repo.get_conversation_for_user(conversation_id, user_id)
        if conversation is None:
            raise NotFoundException("Chat conversation not found", code="CONVERSATION_NOT_FOUND")

        self.repo.delete_conversation(conversation)
        self.db.commit()
        return True
