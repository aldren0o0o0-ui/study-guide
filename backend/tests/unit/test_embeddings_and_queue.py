import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors
from rq.timeouts import TimerDeathPenalty

from app.core.config import Settings
from app.modules.processing.embeddings import (
    GeminiEmbeddingProvider,
    validate_embedding_vector,
)
from app.modules.processing.exceptions import (
    DocumentProcessingError,
    EmbeddingPermanentError,
    EmbeddingTransientError,
)
from app.modules.processing.queue import RQProcessingQueue
from app.modules.processing.worker import WindowsSimpleWorker


@dataclass
class _FakeEmbeddingItem:
    values: list[float]


@dataclass
class _FakeEmbedResponse:
    embeddings: list[_FakeEmbeddingItem] | None


class _FakeModelsClient:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._handler(model=model, contents=contents, config=config)


class _FakeGenAIClient:
    def __init__(self, handler: Any) -> None:
        self.models = _FakeModelsClient(handler)


def test_gemini_embedding_provider_batches_and_validates_dimensions() -> None:
    def _handler(*, model: str, contents: list[str], config: Any) -> _FakeEmbedResponse:
        _ = (model, config)
        return _FakeEmbedResponse(
            embeddings=[
                _FakeEmbeddingItem(values=[float(i + 1), 0.5, -0.25, 0.125])
                for i in range(len(contents))
            ]
        )

    fake_client = _FakeGenAIClient(_handler)
    provider = GeminiEmbeddingProvider(
        api_key="test-key-not-logged",
        model_name="gemini-embedding-001",
        dimension=4,
        batch_size=2,
        genai_client=fake_client,
    )

    doc_vectors = provider.embed_documents(["Chunk A", "Chunk B", "Chunk C"])
    assert len(doc_vectors) == 3
    assert all(len(v) == 4 for v in doc_vectors)
    # 3 chunks with batch_size=2 -> 2 provider calls
    assert len(fake_client.models.calls) == 2

    query_vec = provider.embed_query("What is mitosis?")
    assert len(query_vec) == 4
    assert len(fake_client.models.calls) == 3


def test_gemini_embedding_provider_handles_rate_limits_transient_and_permanent_errors() -> None:
    sleep_delays: list[float] = []
    attempts = 0

    def _flaky_handler(*, model: str, contents: list[str], config: Any) -> _FakeEmbedResponse:
        _ = (model, config)
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise genai_errors.APIError(429, {"error": {"message": "Rate limit exceeded"}})
        return _FakeEmbedResponse(
            embeddings=[_FakeEmbeddingItem(values=[0.1, 0.2, 0.3]) for _ in contents]
        )

    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimension=3,
        max_retries=3,
        initial_backoff_seconds=0.25,
        sleep_fn=sleep_delays.append,
        genai_client=_FakeGenAIClient(_flaky_handler),
    )

    vec = provider.embed_query("Retryable query")
    assert vec == [0.1, 0.2, 0.3]
    assert attempts == 3
    assert sleep_delays == [0.25, 0.5]

    # Exhausting retries on 429 raises EmbeddingTransientError (EMBEDDING_RATE_LIMITED)
    always_429_provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimension=3,
        max_retries=2,
        sleep_fn=lambda _: None,
        genai_client=_FakeGenAIClient(
            lambda **_: (_ for _ in ()).throw(
                genai_errors.APIError(429, {"error": {"message": "Quota exceeded"}})
            )
        ),
    )
    with pytest.raises(EmbeddingTransientError) as exc_info:
        always_429_provider.embed_query("Rate limited query")
    assert exc_info.value.code == "EMBEDDING_RATE_LIMITED"
    assert exc_info.value.retryable is True

    # Non-retryable 400 error fails immediately on first attempt without retrying
    perm_attempts = 0

    def _perm_fail_handler(**_: Any) -> Any:
        nonlocal perm_attempts
        perm_attempts += 1
        raise genai_errors.APIError(400, {"error": {"message": "Invalid argument"}})

    perm_provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimension=3,
        max_retries=3,
        sleep_fn=lambda _: None,
        genai_client=_FakeGenAIClient(_perm_fail_handler),
    )
    with pytest.raises(EmbeddingPermanentError) as perm_exc:
        perm_provider.embed_query("Bad request")
    assert perm_exc.value.code == "EMBEDDING_PROVIDER_REJECTED"
    assert perm_attempts == 1


def test_embedding_dimension_and_response_validation() -> None:
    # Missing API key
    no_key_provider = GeminiEmbeddingProvider(api_key=None, dimension=4)
    with pytest.raises(EmbeddingPermanentError) as exc_info:
        no_key_provider.embed_query("Query without key")
    assert exc_info.value.code == "EMBEDDING_API_KEY_MISSING"

    # Dimension mismatch
    with pytest.raises(EmbeddingPermanentError) as exc_info:
        validate_embedding_vector([0.1, 0.2], expected_dim=4)
    assert exc_info.value.code == "EMBEDDING_DIMENSION_MISMATCH"

    # NaN / Inf values rejected
    with pytest.raises(EmbeddingPermanentError) as exc_info:
        validate_embedding_vector([float("nan"), 0.1, 0.2, 0.3], expected_dim=4)
    assert exc_info.value.code == "INVALID_EMBEDDING_RESPONSE"

    # Empty response from provider rejected
    bad_resp_provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimension=3,
        genai_client=_FakeGenAIClient(lambda **_: _FakeEmbedResponse(embeddings=None)),
    )
    with pytest.raises(EmbeddingPermanentError) as exc_info:
        bad_resp_provider.embed_documents(["Test chunk"])
    assert exc_info.value.code == "INVALID_EMBEDDING_RESPONSE"


def test_rq_processing_queue_and_windows_worker_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert WindowsSimpleWorker.death_penalty_class is TimerDeathPenalty

    doc_id = uuid.uuid4()
    fake_redis = MagicMock()
    enqueued_calls: list[dict[str, Any]] = []

    class _FakeJob:
        def __init__(self, job_id: str) -> None:
            self.id = job_id

    class _FakeQueue:
        def __init__(self, name: str, connection: Any, default_timeout: int) -> None:
            self.name = name
            self.connection = connection
            self.default_timeout = default_timeout

        def enqueue_call(
            self,
            *,
            func: Any,
            args: tuple[Any, ...],
            job_id: str,
            timeout: int,
        ) -> _FakeJob:
            enqueued_calls.append(
                {"func": func, "args": args, "job_id": job_id, "timeout": timeout}
            )
            return _FakeJob(job_id)

    monkeypatch.setattr("app.modules.processing.queue.rq.Queue", _FakeQueue)

    queue_service = RQProcessingQueue(
        settings=Settings(REDIS_URL="redis://localhost:6379/0", PROCESSING_EAGER_FALLBACK=False),
        redis_connection=fake_redis,
    )
    job_id = queue_service.enqueue_document(doc_id)
    assert job_id == f"doc-process-{doc_id}"
    assert len(enqueued_calls) == 1
    assert enqueued_calls[0]["args"] == (str(doc_id),)

    # When Redis raises ConnectionError and eager fallback is False -> raises QUEUE_UNAVAILABLE
    class _FailingQueue:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("Redis offline")

    monkeypatch.setattr("app.modules.processing.queue.rq.Queue", _FailingQueue)
    with pytest.raises(DocumentProcessingError) as exc_info:
        queue_service.enqueue_document(doc_id)
    assert exc_info.value.code == "QUEUE_UNAVAILABLE"
    assert exc_info.value.retryable is True
