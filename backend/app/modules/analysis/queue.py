import logging
import threading
import uuid

import redis
import rq
from rq.job import Job

from app.core.config import Settings, get_settings
from app.database.session import get_session_factory
from app.modules.analysis.exceptions import AnalysisGenerationError
from app.modules.analysis.service import AnalysisExecutionService, AnalysisQueueDispatcher

logger = logging.getLogger("app.modules.analysis.queue")


def execute_analysis_job(analysis_id_str: str) -> str:
    """
    RQ background job entrypoint for generating a document analysis.
    """
    analysis_id = uuid.UUID(str(analysis_id_str))
    session_factory = get_session_factory()
    db = session_factory()
    try:
        service = AnalysisExecutionService(db)
        analysis = service.execute_analysis(analysis_id)
        if analysis is None:
            return "deleted"
        return analysis.status
    finally:
        db.close()


class RQAnalysisQueue:
    """Redis + RQ implementation of `AnalysisQueueDispatcher`."""

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

    def enqueue_analysis(self, analysis_id: uuid.UUID) -> str:
        job_id = f"analysis-generate-{analysis_id}"
        try:
            conn = self._get_redis()

            # If fallback is enabled and no workers are listening on this queue,
            # execute via background thread to prevent indefinite stalling in Redis.
            if self.settings.PROCESSING_EAGER_FALLBACK and not self._has_active_workers(conn):
                logger.info(
                    "No active RQ workers listening; executing analysis in background thread",
                    extra={
                        "extra_data": {
                            "analysis_id": str(analysis_id),
                            "queue": self.settings.RQ_QUEUE_NAME,
                        }
                    },
                )
                thread = threading.Thread(
                    target=execute_analysis_job,
                    args=(str(analysis_id),),
                    daemon=True,
                    name=f"worker-thread-analysis-{analysis_id}",
                )
                thread.start()
                return f"thread-{analysis_id}"

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
                        "Analysis generation job is already active in RQ",
                        extra={
                            "extra_data": {
                                "analysis_id": str(analysis_id),
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
                func=execute_analysis_job,
                args=(str(analysis_id),),
                job_id=job_id,
                timeout=self.settings.PROCESSING_JOB_TIMEOUT_SECONDS,
            )
            logger.info(
                "Enqueued analysis generation job",
                extra={
                    "extra_data": {
                        "analysis_id": str(analysis_id),
                        "job_id": str(job.id),
                        "queue": self.settings.RQ_QUEUE_NAME,
                    }
                },
            )
            return str(job.id)
        except Exception as exc:
            if self.settings.PROCESSING_EAGER_FALLBACK:
                logger.warning(
                    "Redis unavailable; executing analysis generation via background thread",
                    extra={
                        "extra_data": {
                            "analysis_id": str(analysis_id),
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                thread = threading.Thread(
                    target=execute_analysis_job,
                    args=(str(analysis_id),),
                    daemon=True,
                    name=f"worker-thread-analysis-{analysis_id}",
                )
                thread.start()
                return f"thread-{analysis_id}"

            logger.warning(
                "Failed to enqueue analysis job in Redis/RQ",
                extra={
                    "extra_data": {
                        "analysis_id": str(analysis_id),
                        "error_type": type(exc).__name__,
                    }
                },
            )
            raise AnalysisGenerationError(
                f"Failed to enqueue background analysis job: {type(exc).__name__}",
                code="QUEUE_UNAVAILABLE",
                user_message=(
                    "Background analysis queue is temporarily unavailable. "
                    "Please tap Retry Analysis once the worker service is online."
                ),
                retryable=True,
            ) from exc


def create_analysis_queue(settings: Settings | None = None) -> AnalysisQueueDispatcher:
    return RQAnalysisQueue(settings=settings)


def get_analysis_queue() -> AnalysisQueueDispatcher:
    return create_analysis_queue()
