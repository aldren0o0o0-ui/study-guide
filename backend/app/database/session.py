import logging
from collections.abc import Generator
from typing import Literal

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger("app.database")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _register_models() -> None:
    import app.modules.analysis.models  # noqa: F401
    import app.modules.auth.models  # noqa: F401
    import app.modules.chat.models  # noqa: F401
    import app.modules.documents.models  # noqa: F401
    import app.modules.exams.models  # noqa: F401
    import app.modules.voice_practice.models  # noqa: F401


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _register_models()
        settings = get_settings()
        _engine = create_engine(
            settings.DATABASE_URL,
            echo=settings.DB_ECHO,
            pool_pre_ping=True,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def check_database_status() -> Literal["ok", "unavailable"]:
    """Non-throwing check for database connectivity."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        logger.warning(
            "Database connectivity check failed",
            extra={"extra_data": {"error_type": type(exc).__name__}},
        )
        return "unavailable"


def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None
