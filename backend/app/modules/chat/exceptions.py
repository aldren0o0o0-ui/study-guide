class ChatError(Exception):
    """Base exception for AI chat errors with a safe user-facing message."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CHAT_ERROR",
        user_message: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.user_message = (
            user_message
            or "Could not process your chat message. Please try again."
        )
        self.retryable = retryable


class ChatTransientError(ChatError):
    """Retryable AI chat generation failure (e.g. rate limit, network timeout, 5xx)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CHAT_TRANSIENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "The AI chat service is temporarily busy. Please retry in a moment.",
            retryable=True,
        )


class ChatPermanentError(ChatError):
    """Non-retryable AI chat failure (e.g. missing API key, invalid configuration)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CHAT_PERMANENT_ERROR",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            user_message=user_message
            or "The AI chat service could not generate an answer for this request.",
            retryable=False,
        )

