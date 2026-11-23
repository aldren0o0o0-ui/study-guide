"""
Windows-compatible RQ worker entrypoint for StudyGuide document processing.

Usage (Windows PowerShell, no Docker required):
    Push-Location d:\\study-guide\\backend
    .\\venv\\Scripts\\python.exe -m app.modules.processing.worker
"""

import argparse
import logging
import os

import redis
import rq
from rq.timeouts import TimerDeathPenalty

from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger("app.modules.processing.worker")


class WindowsSimpleWorker(rq.SimpleWorker):
    """
    RQ worker compatible with native Windows Python processes.

    Uses `SimpleWorker` (in-process job execution without `os.fork()`) and
    `TimerDeathPenalty` (thread-timer timeout instead of POSIX `signal.SIGALRM`).
    """

    death_penalty_class = TimerDeathPenalty


def run_worker(*, burst: bool = False) -> None:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    conn = redis.Redis.from_url(settings.REDIS_URL)
    queue = rq.Queue(
        name=settings.RQ_QUEUE_NAME,
        connection=conn,
        default_timeout=settings.PROCESSING_JOB_TIMEOUT_SECONDS,
    )

    worker_class = WindowsSimpleWorker if os.name == "nt" else rq.SimpleWorker
    worker = worker_class([queue], connection=conn)
    logger.info(
        "Starting StudyGuide RQ document processing worker",
        extra={
            "extra_data": {
                "queue": settings.RQ_QUEUE_NAME,
                "burst": burst,
                "worker_class": worker_class.__name__,
            }
        },
    )
    worker.work(burst=burst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the StudyGuide RQ document processing worker."
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Run in burst mode (process all currently queued jobs and exit).",
    )
    args = parser.parse_args()
    run_worker(burst=bool(args.burst))


if __name__ == "__main__":
    main()
