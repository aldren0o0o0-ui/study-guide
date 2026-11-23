import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentUpdateRequest(BaseModel):
    original_filename: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("original_filename", mode="before")
    @classmethod
    def strip_filename(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    original_filename: str
    file_extension: str
    mime_type: str
    file_size_bytes: int
    checksum_sha256: str
    processing_status: str
    processing_error_code: str | None = None
    processing_error_message: str | None = None
    processed_at: datetime | None = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentProcessingStatusRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    processing_status: str
    processing_error_code: str | None = None
    processing_error_message: str | None = None
    processed_at: datetime | None = None
    chunk_count: int = 0
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentRead]
    total: int
    limit: int
    offset: int


class DocumentDeleteResponse(BaseModel):
    deleted: bool


class DocumentAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    user_id: uuid.UUID
    asset_type: str
    mime_type: str
    file_size_bytes: int
    page_number: int | None = None
    section_title: str | None = None
    caption: str | None = None
    order_index: int
    width: int | None = None
    height: int | None = None
    created_at: datetime

