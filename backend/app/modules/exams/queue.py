import logging
import threading
import uuid
from typing import Any

import redis
import rq
from rq.job import Job

from app.core.config import Settings, get_settings
from app.database.session import get_session_factory
from app.modules.exams.exceptions import ExamError
from app.modules.exams.service import ExamExecutionService, ExamQueueDispatcher

logger = logging.getLogger("app.modules.exams.queue")


def execute_exam_generation_job(reviewer_id_str: str) -> None:
    """
    Top-level RQ job function executed by the worker to generate an exam reviewer.
    """
    reviewer_id = uuid.UUID(reviewer_id_str)
    session_factory = get_session_factory()
    db = session_factory()
    try:
        service = ExamExecutionService(db)
        service.execute_reviewer_generation(reviewer_id)
    except Exception as exc:
        logger.error(
            "Unhandled exception in exam generation worker job",
            extra={
                "extra_data": {
                    "reviewer_id": reviewer_id_str,
                    "error_type": type(exc).__name__,
                }
            },
            exc_info=True,
        )
        raise
    finally:
        db.close()


class RQExamQueue:
    """
    Redis / RQ dispatcher for exam reviewer generation.
    Supports `PROCESSING_EAGER_FALLBACK` when Redis is offline.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        redis_connection: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._redis = redis_connection

    def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        self._redis = redis.Redis.from_url(
            self.settings.REDIS_URL,
            socket_connect_timeout=2.0,
            socket_timeout=5.0,
        )
        return self._redis

    def _has_active_workers(self, conn: Any) -> bool:
        try:
            workers = rq.Worker.all(connection=conn)
            queue_name = self.settings.RQ_QUEUE_NAME
            return any(queue_name in [q.name for q in w.queues] for w in workers)
        except Exception:
            return False

    def enqueue_exam(self, reviewer_id: uuid.UUID) -> str:
        job_id = f"exam-generate-{reviewer_id}"
        try:
            conn = self._get_redis()

            # If fallback is enabled and no workers are listening on this queue,
            # execute via background thread to prevent indefinite stalling in Redis.
            if self.settings.PROCESSING_EAGER_FALLBACK and not self._has_active_workers(conn):
                logger.info(
                    "No active RQ workers listening; executing exam in background thread",
                    extra={
                        "extra_data": {
                            "reviewer_id": str(reviewer_id),
                            "queue": self.settings.RQ_QUEUE_NAME,
                        }
                    },
                )
                thread = threading.Thread(
                    target=execute_exam_generation_job,
                    args=(str(reviewer_id),),
                    daemon=True,
                    name=f"worker-thread-exam-{reviewer_id}",
                )
                thread.start()
                return f"thread-{reviewer_id}"

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
                        "Exam generation job is already active in RQ",
                        extra={
                            "extra_data": {
                                "reviewer_id": str(reviewer_id),
                                "job_id": job_id,
                                "status": str(status),
                            }
                        },
                    )
                    return job_id
                existing_job.delete()
            except Exception:
                pass

            job = queue.enqueue_call(
                func=execute_exam_generation_job,
                args=(str(reviewer_id),),
                job_id=job_id,
                timeout=self.settings.PROCESSING_JOB_TIMEOUT_SECONDS,
            )
            logger.info(
                "Enqueued exam generation job",
                extra={
                    "extra_data": {
                        "reviewer_id": str(reviewer_id),
                        "job_id": str(job.id),
                        "queue": self.settings.RQ_QUEUE_NAME,
                    }
                },
            )
            return str(job.id)
        except Exception as exc:
            if self.settings.PROCESSING_EAGER_FALLBACK:
                logger.warning(
                    "Redis unavailable; executing exam generation via background thread",
                    extra={
                        "extra_data": {
                            "reviewer_id": str(reviewer_id),
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                thread = threading.Thread(
                    target=execute_exam_generation_job,
                    args=(str(reviewer_id),),
                    daemon=True,
                    name=f"worker-thread-exam-{reviewer_id}",
                )
                thread.start()
                return f"thread-{reviewer_id}"

            logger.warning(
                "Failed to enqueue exam job in Redis/RQ",
                extra={
                    "extra_data": {
                        "reviewer_id": str(reviewer_id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            raise ExamError(
                f"Failed to enqueue background exam job: {type(exc).__name__}",
                code="QUEUE_UNAVAILABLE",
                user_message=(
                    "Background exam queue is temporarily unavailable. "
                    "Please tap Retry once the worker service is online."
                ),
                retryable=True,
            ) from exc


def create_exam_queue(settings: Settings | None = None) -> ExamQueueDispatcher:
    return RQExamQueue(settings=settings)


def get_exam_queue() -> ExamQueueDispatcher:
    return create_exam_queue()
