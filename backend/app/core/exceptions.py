import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_request_id
from app.shared.responses import ErrorPayload, ErrorResponse, FieldErrorDetail

logger = logging.getLogger("app.exceptions")


class AppException(Exception):
    """Base application exception producing a consistent ErrorResponse."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "APP_ERROR",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: list[FieldErrorDetail] | dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class BadRequestException(AppException):
    def __init__(
        self,
        message: str = "Bad request",
        *,
        code: str = "BAD_REQUEST",
        details: list[FieldErrorDetail] | dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class UnauthorizedException(AppException):
    def __init__(
        self,
        message: str = "Authentication required",
        *,
        code: str = "UNAUTHORIZED",
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class ForbiddenException(AppException):
    def __init__(
        self,
        message: str = "You do not have permission to perform this action",
        *,
        code: str = "FORBIDDEN",
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class NotFoundException(AppException):
    def __init__(
        self,
        message: str = "Requested resource was not found",
        *,
        code: str = "NOT_FOUND",
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ConflictException(AppException):
    def __init__(
        self,
        message: str = "Resource conflict",
        *,
        code: str = "CONFLICT",
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_409_CONFLICT,
        )


class PayloadTooLargeException(AppException):
    def __init__(
        self,
        message: str = "Uploaded file exceeds maximum allowed size",
        *,
        code: str = "FILE_TOO_LARGE",
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )


class ServiceUnavailableException(AppException):
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        *,
        code: str = "SERVICE_UNAVAILABLE",
    ) -> None:
        super().__init__(
            message,
            code=code,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


def _build_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[FieldErrorDetail] | dict[str, object] | None = None,
) -> JSONResponse:
    request_id = get_request_id()
    body = ErrorResponse(
        error=ErrorPayload(
            code=code,
            message=message,
            details=details,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def handle_app_exception(_request: Request, exc: AppException) -> JSONResponse:
        return _build_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details: list[FieldErrorDetail] = []
        for err in exc.errors():
            loc_parts = [str(part) for part in err.get("loc", ()) if part != "body"]
            field_path = ".".join(loc_parts) if loc_parts else "request"
            details.append(
                FieldErrorDetail(
                    field=field_path,
                    message=str(err.get("msg", "Invalid value")),
                    code=str(err.get("type", "validation_error")),
                )
            )
        return _build_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=details,
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        detail_msg = exc.detail if isinstance(exc.detail, str) else "HTTP error occurred"
        return _build_error_response(
            status_code=exc.status_code,
            code=f"HTTP_{exc.status_code}",
            message=detail_msg,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled server error",
            extra={"extra_data": {"path": request.url.path, "error_type": type(exc).__name__}},
        )
        return _build_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again later.",
        )
