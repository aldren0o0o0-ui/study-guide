import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatSourceCitation(BaseModel):
    """
    Detailed source attribution for a grounded chat statement.
    """

    model_config = ConfigDict(from_attributes=True)

    source_key: str = Field(..., min_length=1, max_length=16)
    document_id: uuid.UUID
    original_filename: str
    chunk_id: uuid.UUID
    chunk_index: int
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    excerpt: str


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    citations: list[ChatSourceCitation] | None = None
    created_at: datetime


class ChatConversationDocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    original_filename: str
    file_extension: str
    processing_status: str


class ChatConversationSummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    documents_count: int = 0
    messages_count: int = 0
    latest_message_preview: str | None = None
    source_documents: list[ChatConversationDocumentRead] = Field(default_factory=list)


class ChatConversationDetailRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    source_documents: list[ChatConversationDocumentRead] = Field(default_factory=list)
    messages: list[ChatMessageRead] = Field(default_factory=list)


class ChatConversationListResponse(BaseModel):
    items: list[ChatConversationSummaryRead]
    total: int
    limit: int
    offset: int


class ChatConversationCreateRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=10)
    title: str | None = Field(default=None, max_length=255)
    initial_message: str | None = Field(default=None, max_length=4000)

    @field_validator("document_ids")
    @classmethod
    def deduplicate_document_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        unique_ids = list(dict.fromkeys(value))
        if not unique_ids:
            raise ValueError("At least one document must be selected.")
        return unique_ids

    @field_validator("title", "initial_message", mode="before")
    @classmethod
    def normalize_strings(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return value


class ChatMessageSendRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Chat message cannot be empty.")
        return stripped


class ChatMessageSendResponse(BaseModel):
    user_message: ChatMessageRead
    assistant_message: ChatMessageRead
    conversation_title: str | None = None


class ChatConversationDeleteResponse(BaseModel):
    deleted: bool
    conversation_id: uuid.UUID
