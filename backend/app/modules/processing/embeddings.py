import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import httpcore
import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import Settings, get_settings
from app.modules.processing.exceptions import (
    EmbeddingPermanentError,
    EmbeddingTransientError,
)

logger = logging.getLogger("app.modules.processing.embeddings")

EmbeddingTaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_NETWORK_ERRORS = (
    TimeoutError,
    ConnectionError,
    OSError,
    httpx.HTTPError,
    httpcore.NetworkError,
    httpcore.ProtocolError,
)


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


def validate_embedding_vector(vector: Sequence[float] | None, *, expected_dim: int) -> list[float]:
    if not vector:
        raise EmbeddingPermanentError(
            "Embedding provider returned an empty vector",
            code="INVALID_EMBEDDING_RESPONSE",
            user_message="The AI embedding provider returned an invalid response.",
        )
    if len(vector) != expected_dim:
        raise EmbeddingPermanentError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(vector)}",
            code="EMBEDDING_DIMENSION_MISMATCH",
            user_message="The AI embedding provider returned an unexpected vector dimension.",
        )

    normalized: list[float] = []
    for val in vector:
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise EmbeddingPermanentError(
                "Embedding vector contains non-numeric values",
                code="INVALID_EMBEDDING_RESPONSE",
            ) from exc
        if not math.isfinite(fval):
            raise EmbeddingPermanentError(
                "Embedding vector contains NaN or infinite values",
                code="INVALID_EMBEDDING_RESPONSE",
            )
        normalized.append(fval)

    return normalized


class GeminiEmbeddingProvider:
    """
    Gemini embedding provider backed by `google-genai` (`client.models.embed_content`).

    Never logs API keys, document text, or query text.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = "gemini-embedding-001",
        dimension: int = 768,
        batch_size: int = 32,
        max_retries: int = 3,
        initial_backoff_seconds: float = 0.5,
        sleep_fn: Callable[[float], None] = time.sleep,
        genai_client: object | None = None,
    ) -> None:
        self._api_key = api_key.strip() if api_key else ""
        self.model_name = model_name
        self._dimension = dimension
        self.batch_size = max(1, batch_size)
        self.max_retries = max(1, max_retries)
        self.initial_backoff_seconds = max(0.0, initial_backoff_seconds)
        self._sleep_fn = sleep_fn
        self._client = genai_client

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise EmbeddingPermanentError(
                "GEMINI_API_KEY is not configured",
                code="EMBEDDING_API_KEY_MISSING",
                user_message=(
                    "AI document processing is not configured on the server (missing API key)."
                ),
            )
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=60_000),
        )
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        cleaned_texts: list[str] = []
        for idx, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                raise EmbeddingPermanentError(
                    f"Cannot embed empty document chunk at index {idx}",
                    code="EMPTY_CHUNK_TEXT",
                )
            cleaned_texts.append(stripped)

        results: list[list[float]] = []
        for start in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[start : start + self.batch_size]
            batch_vectors = self._embed_batch_with_retry(
                batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
            results.extend(batch_vectors)

        return results

    def embed_query(self, query: str) -> list[float]:
        stripped = query.strip()
        if not stripped:
            raise EmbeddingPermanentError(
                "Search query cannot be empty",
                code="EMPTY_QUERY_TEXT",
                user_message="Please enter a non-empty search query.",
            )
        vectors = self._embed_batch_with_retry(
            [stripped],
            task_type="RETRIEVAL_QUERY",
        )
        return vectors[0]

    def _embed_batch_with_retry(
        self,
        batch: list[str],
        *,
        task_type: EmbeddingTaskType,
    ) -> list[list[float]]:
        attempt = 0
        last_transient_error: EmbeddingTransientError | None = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                return self._call_provider(batch, task_type=task_type)
            except EmbeddingPermanentError:
                raise
            except EmbeddingTransientError as exc:
                last_transient_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.initial_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Transient embedding error; retrying batch",
                    extra={
                        "extra_data": {
                            "attempt": attempt,
                            "max_retries": self.max_retries,
                            "error_code": exc.code,
                        }
                    },
                )
                self._sleep_fn(delay)

        assert last_transient_error is not None
        raise last_transient_error

    def _call_provider(
        self,
        batch: list[str],
        *,
        task_type: EmbeddingTaskType,
    ) -> list[list[float]]:
        client = self._get_client()
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimension,
        )

        try:
            response = client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=config,
            )
        except (EmbeddingPermanentError, EmbeddingTransientError):
            raise
        except genai_errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            if status_code == 429:
                raise EmbeddingTransientError(
                    "Gemini embedding rate limit exceeded",
                    code="EMBEDDING_RATE_LIMITED",
                    user_message=(
                        "AI rate limit reached while processing document. "
                        "Please retry processing in a moment."
                    ),
                ) from exc
            if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES:
                raise EmbeddingTransientError(
                    f"Gemini API transient HTTP {status_code} error",
                    code="EMBEDDING_PROVIDER_UNAVAILABLE",
                ) from exc
            raise EmbeddingPermanentError(
                f"Gemini API rejected embedding request (status={status_code})",
                code="EMBEDDING_PROVIDER_REJECTED",
                user_message=(
                    "The AI embedding service rejected the request. "
                    "Please check server configuration."
                ),
            ) from exc
        except _NETWORK_ERRORS as exc:
            raise EmbeddingTransientError(
                f"Network error calling Gemini embedding provider: {type(exc).__name__}",
                code="EMBEDDING_NETWORK_ERROR",
            ) from exc
        except Exception as exc:
            raise EmbeddingPermanentError(
                f"Unexpected error from Gemini embedding provider: {type(exc).__name__}",
                code="EMBEDDING_UNEXPECTED_ERROR",
            ) from exc

        raw_embeddings = getattr(response, "embeddings", None)
        if not raw_embeddings or not isinstance(raw_embeddings, Sequence):
            raise EmbeddingPermanentError(
                "Gemini response did not contain an 'embeddings' list",
                code="INVALID_EMBEDDING_RESPONSE",
            )

        if len(raw_embeddings) != len(batch):
            raise EmbeddingPermanentError(
                f"Expected {len(batch)} embeddings from provider, got {len(raw_embeddings)}",
                code="INVALID_EMBEDDING_RESPONSE",
            )

        validated: list[list[float]] = []
        for item in raw_embeddings:
            values = getattr(item, "values", None)
            validated.append(validate_embedding_vector(values, expected_dim=self._dimension))

        return validated


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    cfg = settings or get_settings()
    return GeminiEmbeddingProvider(
        api_key=cfg.GEMINI_API_KEY,
        model_name=cfg.GEMINI_EMBEDDING_MODEL,
        dimension=cfg.GEMINI_EMBEDDING_DIMENSION,
        batch_size=cfg.GEMINI_EMBEDDING_BATCH_SIZE,
    )


def get_embedding_provider() -> EmbeddingProvider:
    return create_embedding_provider()
