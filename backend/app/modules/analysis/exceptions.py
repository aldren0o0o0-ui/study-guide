class AnalysisGenerationError(Exception):
    """Base exception for in-depth analysis generation errors with a safe user-facing message."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ANALYSIS_GENERATION_FAILED",
        user_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.user_message = (
            user_message
            or "Could not generate the in-depth analysis. Please retry or try again later."
        )
        self.retryable = retryable


class AnalysisTransientError(AnalysisGenerationError):
    """Retryable AI generation failure (e.g., provider rate limit, 5xx, or network timeout)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ANALYSIS_TRANSIENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "The AI analysis service is temporarily busy. Please tap Retry Analysis shortly.",
            retryable=True,
        )


class AnalysisPermanentError(AnalysisGenerationError):
    """Non-retryable AI generation failure (e.g., invalid configuration or malformed output)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ANALYSIS_PERMANENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "The AI service could not produce a valid structured analysis for this request.",
            retryable=False,
        )
