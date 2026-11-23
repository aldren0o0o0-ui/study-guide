class DocumentProcessingError(Exception):
    """Base exception for document processing pipeline errors with a safe user-facing message."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "PROCESSING_FAILED",
        user_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.user_message = (
            user_message or "Document processing failed. Please try again or upload another file."
        )
        self.retryable = retryable


class DocumentExtractionError(DocumentProcessingError):
    """Raised when text cannot be extracted from an uploaded document."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EXTRACTION_FAILED",
        user_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message or "Could not extract readable text from this document.",
            retryable=retryable,
        )


class EmbeddingError(DocumentProcessingError):
    """Base error for embedding provider failures."""


class EmbeddingTransientError(EmbeddingError):
    """Retryable embedding failure (e.g., provider rate limit, 5xx, or network timeout)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EMBEDDING_TRANSIENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "The AI embedding service is temporarily busy. Please retry processing shortly.",
            retryable=True,
        )


class EmbeddingPermanentError(EmbeddingError):
    """Non-retryable embedding failure (e.g., invalid configuration or dimension mismatch)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EMBEDDING_PERMANENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "Document embeddings could not be generated due to a provider error.",
            retryable=False,
        )
