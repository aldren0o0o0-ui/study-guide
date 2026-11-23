import logging
import threading
import uuid
from typing import Protocol

import redis
import rq
from rq.job import Job

from app.core.config import Settings, get_settings
from app.database.session import get_session_factory
from app.modules.processing.exceptions import DocumentProcessingError
from app.modules.processing.service import DocumentProcessingService

logger = logging.getLogger("app.modules.processing.queue")


class ProcessingQueue(Protocol):
    def enqueue_document(self, document_id: uuid.UUID) -> str: ...


def process_document_job(document_id_str: str) -> str:
    """
    RQ background job entrypoint for processing an uploaded document.

    Creates its own database session and ensures deterministic cleanup.
    """
    doc_id = uuid.UUID(str(document_id_str))
    session_factory = get_session_factory()
    db = session_factory()
    try:
        service = DocumentProcessingService(db)
        doc = service.process_document(doc_id)
        if doc is None:
            return "deleted"
        return doc.processing_status
    finally:
        db.close()


class RQProcessingQueue:
    """Redis + RQ implementation of `ProcessingQueue`."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        redis_connection: redis.Redis | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._redis = redis_connection

    def _get_redis(self) -> redis.Redis:
        if self._redis is not None:
            return self._redis
        self._redis = redis.Redis.from_url(
            self.settings.REDIS_URL,
            socket_connect_timeout=2.0,
            socket_timeout=5.0,
        )
        return self._redis

    def _has_active_workers(self, conn: redis.Redis) -> bool:
        try:
            workers = rq.Worker.all(connection=conn)
            queue_name = self.settings.RQ_QUEUE_NAME
            return any(queue_name in [q.name for q in w.queues] for w in workers)
        except Exception:
            return False

    def enqueue_document(self, document_id: uuid.UUID) -> str:
        job_id = f"doc-process-{document_id}"
        try:
            conn = self._get_redis()

            # If fallback is enabled and no workers are listening on this queue,
            # execute via background thread to prevent indefinite stalling in Redis.
            if self.settings.PROCESSING_EAGER_FALLBACK and not self._has_active_workers(conn):
                logger.info(
                    "No active RQ workers listening; executing processing in background thread",
                    extra={
                        "extra_data": {
                            "document_id": str(document_id),
                            "queue": self.settings.RQ_QUEUE_NAME,
                        }
                    },
                )
                thread = threading.Thread(
                    target=process_document_job,
                    args=(str(document_id),),
                    daemon=True,
                    name=f"worker-thread-doc-{document_id}",
                )
                thread.start()
                return f"thread-{document_id}"

            queue = rq.Queue(
                name=self.settings.RQ_QUEUE_NAME,
                connection=conn,
                default_timeout=self.settings.PROCESSING_JOB_TIMEOUT_SECONDS,
            )

            try:
                existing_job = Job.fetch(job_id, connection=conn)
                status = existing_job.get_status(refresh=True)
                if status in ("queued", "started", "deferred", "scheduled"):
                    logger.info(
                        "Document processing job is already active in RQ",
                        extra={
                            "extra_data": {
                                "document_id": str(document_id),
                                "job_id": job_id,
                                "status": str(status),
                            }
                        },
                    )
                    return job_id
                existing_job.delete()
            except Exception:
                # Job does not exist in Redis yet
                pass

            job = queue.enqueue_call(
                func=process_document_job,
                args=(str(document_id),),
                job_id=job_id,
                timeout=self.settings.PROCESSING_JOB_TIMEOUT_SECONDS,
            )
            logger.info(
                "Enqueued document processing job",
                extra={
                    "extra_data": {
                        "document_id": str(document_id),
                        "job_id": str(job.id),
                        "queue": self.settings.RQ_QUEUE_NAME,
                    }
                },
            )
            return str(job.id)
        except Exception as exc:
            if self.settings.PROCESSING_EAGER_FALLBACK:
                logger.warning(
                    "Redis unavailable; executing document processing via background thread",
                    extra={
                        "extra_data": {
                            "document_id": str(document_id),
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                thread = threading.Thread(
                    target=process_document_job,
                    args=(str(document_id),),
                    daemon=True,
                    name=f"worker-thread-doc-{document_id}",
                )
                thread.start()
                return f"thread-{document_id}"

            logger.warning(
                "Failed to enqueue document processing job in Redis/RQ",
                extra={
                    "extra_data": {
                        "document_id": str(document_id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            raise DocumentProcessingError(
                f"Failed to enqueue background processing job: {type(exc).__name__}",
                code="QUEUE_UNAVAILABLE",
                user_message=(
                    "Background processing queue is temporarily unavailable. "
                    "Please tap Retry Processing once the worker service is online."
                ),
                retryable=True,
            ) from exc


def create_processing_queue(settings: Settings | None = None) -> ProcessingQueue:
    return RQProcessingQueue(settings=settings)


def get_processing_queue() -> ProcessingQueue:
    return create_processing_queue()
