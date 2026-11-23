from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from app.core.logging import get_request_id

T = TypeVar("T")


class FieldErrorDetail(BaseModel):
    field: str = Field(..., description="Field path that caused the error")
    message: str = Field(..., description="Human-readable validation message")
    code: str | None = Field(default=None, description="Machine-readable validation code")


class ErrorPayload(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: list[FieldErrorDetail] | dict[str, object] | None = Field(
        default=None,
        description="Optional structured details about the error",
    )
    request_id: str = Field(
        default_factory=get_request_id,
        description="Correlation ID for tracing the request",
    )


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    error: ErrorPayload


class ApiResponse(BaseModel, Generic[T]):
    success: bool = Field(default=True)
    data: T
    message: str | None = Field(default=None)
    request_id: str = Field(default_factory=get_request_id)
