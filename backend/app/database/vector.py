from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Connection, Engine, Float, cast, func, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.types import TypeDecorator, TypeEngine

DEFAULT_EMBEDDING_DIMENSION = 768

_pgvector_enabled_override: bool | None = None


def set_pgvector_enabled_override(enabled: bool | None) -> None:
    """Allows tests to explicitly force pgvector or fallback SQL expression generation."""
    global _pgvector_enabled_override
    _pgvector_enabled_override = enabled


def serialize_pgvector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def deserialize_pgvector_literal(raw: str) -> list[float]:
    inner = raw.strip().lstrip("[").rstrip("]")
    if not inner:
        return []
    return [float(part.strip()) for part in inner.split(",") if part.strip()]


def is_pgvector_available(bind: Connection | Engine | Session) -> bool:
    """Returns True if the PostgreSQL server has the 'vector' extension installed on disk."""
    try:
        if isinstance(bind, Engine):
            with bind.connect() as conn:
                row = conn.execute(
                    text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
                ).scalar()
                return bool(row)
        row = bind.execute(
            text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        ).scalar()
        return bool(row)
    except Exception:
        return False


def is_pgvector_enabled(bind: Connection | Engine | Session) -> bool:
    """Returns True if 'CREATE EXTENSION vector' is currently active in the database."""
    if _pgvector_enabled_override is not None:
        return _pgvector_enabled_override
    try:
        if isinstance(bind, Engine):
            with bind.connect() as conn:
                row = conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                ).scalar()
                return bool(row)
        row = bind.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
        return bool(row)
    except Exception:
        return False


class EmbeddingVector(TypeDecorator[list[float]]):
    """
    SQLAlchemy column type for fixed-dimension embedding vectors.

    Uses `pgvector.sqlalchemy.Vector(dim)` when the `vector` extension is enabled in
    PostgreSQL, and gracefully falls back to `postgresql.ARRAY(Float)` on local Windows
    installations where `vector.dll` has not yet been copied into PostgreSQL's `lib/` folder.
    """

    impl = ARRAY(Float)
    cache_ok = True

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        super().__init__()
        self.dim = dim
        self._pgvector_type = Vector(dim)
        self._array_type: ARRAY[float] = ARRAY(Float)

    def load_dialect_impl(self, dialect: Any) -> TypeEngine[Any]:
        if _pgvector_enabled_override is True:
            return dialect.type_descriptor(self._pgvector_type)  # type: ignore[no-any-return]
        return dialect.type_descriptor(self._array_type)  # type: ignore[no-any-return]

    def process_bind_param(
        self,
        value: Sequence[float] | None,
        dialect: Any,
    ) -> Any:
        _ = dialect
        if value is None:
            return None
        normalized = [float(v) for v in value]
        if len(normalized) != self.dim:
            raise ValueError(
                f"Expected embedding vector of dimension {self.dim}, got {len(normalized)}"
            )
        if _pgvector_enabled_override is True:
            return serialize_pgvector_literal(normalized)
        return normalized

    def process_result_value(
        self,
        value: Any,
        dialect: Any,
    ) -> list[float] | None:
        _ = dialect
        if value is None:
            return None
        if isinstance(value, str):
            return deserialize_pgvector_literal(value)
        return [float(x) for x in value]


def cosine_distance_expression(
    column: Any,
    query_vector: Sequence[float],
    *,
    use_pgvector: bool,
) -> ColumnElement[float]:
    """
    Builds a SQLAlchemy cosine-distance expression (`0.0` = identical, `1.0` = orthogonal).

    - When `use_pgvector=True`: emits `column <=> '[...]'` via `pgvector` operator.
    - When `use_pgvector=False`: calls the PostgreSQL `sg_cosine_distance(float8[], float8[])`
      SQL function so ranking and filtering execute inside PostgreSQL.
    """
    normalized = [float(x) for x in query_vector]
    if use_pgvector:
        rhs = cast(normalized, Vector(len(normalized)))
        expr: ColumnElement[float] = column.op("<=>", return_type=Float)(rhs)
        return expr

    fallback_expr: ColumnElement[float] = func.sg_cosine_distance(
        column,
        cast(normalized, ARRAY(Float)),
    )
    return fallback_expr
