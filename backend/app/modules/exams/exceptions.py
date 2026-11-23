"""
Custom exceptions for the Exam Reviewer module (Phase 6).
"""

from app.core.exceptions import AppException


class ExamError(AppException):
    """Base exception for all exam reviewer errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EXAM_ERROR",
        user_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, code=code)
        self.user_message = user_message or message
        self.retryable = retryable


class ExamTransientError(ExamError):
    """Transient failures such as rate limits or network drops that should be retried."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EXAM_TRANSIENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "The AI reviewer service is temporarily busy. Please retry shortly.",
            retryable=True,
        )


class ExamPermanentError(ExamError):
    """Permanent errors such as missing API key, prompt rejection, or malformed schema."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EXAM_PERMANENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "Exam generation could not be completed with the provided documents.",
            retryable=False,
        )

