import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import update

from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import RequestContextMiddleware, configure_logging
from app.database.session import dispose_engine, get_session_factory
from app.modules.analysis.models import AnalysisStatus, DocumentAnalysis
from app.modules.documents.models import Document, ProcessingStatus
from app.modules.exams.models import ExamReviewer, ExamStatus
from app.modules.voice_practice.service import get_in_memory_voice_session_store

logger = logging.getLogger("app.main")


def _recover_interrupted_background_jobs() -> None:
    """
    When running with in-process thread fallback (`PROCESSING_EAGER_FALLBACK=True`),
    any queued/processing jobs left behind by a prior server restart or sleep will never
    resume on their own. Mark them failed so users see a Retry button instead of an
    infinite loading spinner.
    """
    try:
        session_factory = get_session_factory()
        with session_factory() as db:
            doc_res = db.execute(
                update(Document)
                .where(
                    Document.processing_status.in_(
                        [ProcessingStatus.QUEUED.value, ProcessingStatus.PROCESSING.value]
                    )
                )
                .values(
                    processing_status=ProcessingStatus.FAILED.value,
                    processing_error_code="PROCESSING_INTERRUPTED",
                    processing_error_message=(
                        "Processing was interrupted by a server restart. Please tap Retry Processing."
                    ),
                )
            )
            analysis_res = db.execute(
                update(DocumentAnalysis)
                .where(
                    DocumentAnalysis.status.in_(
                        [AnalysisStatus.QUEUED.value, AnalysisStatus.PROCESSING.value]
                    )
                )
                .values(
                    status=AnalysisStatus.FAILED.value,
                    error_code="PROCESSING_INTERRUPTED",
                    error_message=(
                        "Analysis was interrupted by a server restart. Please tap Retry."
                    ),
                )
            )
            exam_res = db.execute(
                update(ExamReviewer)
                .where(
                    ExamReviewer.status.in_(
                        [ExamStatus.QUEUED.value, ExamStatus.PROCESSING.value]
                    )
                )
                .values(
                    status=ExamStatus.FAILED.value,
                    error_code="PROCESSING_INTERRUPTED",
                    error_message=(
                        "Exam generation was interrupted by a server restart. Please tap Retry."
                    ),
                )
            )
            db.commit()
            recovered = (
                (doc_res.rowcount or 0)
                + (analysis_res.rowcount or 0)
                + (exam_res.rowcount or 0)
            )
            if recovered:
                logger.warning(
                    "Recovered interrupted background jobs on startup",
                    extra={
                        "extra_data": {
                            "documents": doc_res.rowcount or 0,
                            "analyses": analysis_res.rowcount or 0,
                            "exams": exam_res.rowcount or 0,
                        }
                    },
                )
    except Exception as exc:
        logger.warning(
            "Startup background job recovery skipped",
            extra={"extra_data": {"error_type": type(exc).__name__}},
        )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting application",
        extra={
            "extra_data": {
                "app_name": settings.APP_NAME,
                "environment": settings.APP_ENV,
                "version": settings.APP_VERSION,
            }
        },
    )
    if settings.PROCESSING_EAGER_FALLBACK and settings.APP_ENV != "testing":
        _recover_interrupted_background_jobs()
    yield
    logger.info("Shutting down application")
    get_in_memory_voice_session_store().clear()
    dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)

    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    application.add_middleware(RequestContextMiddleware)

    register_exception_handlers(application)
    application.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return application


app = create_app()
