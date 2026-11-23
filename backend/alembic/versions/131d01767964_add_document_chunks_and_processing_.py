"""add_document_chunks_and_processing_fields

Revision ID: 131d01767964
Revises: 7fc869e91b3b
Create Date: 2026-10-30 22:26:16.122621

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from app.database.vector import DEFAULT_EMBEDDING_DIMENSION, EmbeddingVector

# revision identifiers, used by Alembic.
revision: str = "131d01767964"
down_revision: str | None = "7fc869e91b3b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_pgvector_extension() -> bool:
    bind = op.get_bind()
    row = bind.execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
    ).scalar()
    return bool(row)


def upgrade() -> None:
    has_pgvector = _has_pgvector_extension()
    if has_pgvector:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Install PostgreSQL-native cosine distance helper function for float8[] fallback
    op.execute(
        """
        CREATE OR REPLACE FUNCTION sg_cosine_distance(a float8[], b float8[])
        RETURNS float8
        LANGUAGE sql
        IMMUTABLE STRICT PARALLEL SAFE
        AS $$
            SELECT CASE
                WHEN norm_a = 0 OR norm_b = 0 THEN 1.0
                ELSE GREATEST(0.0, LEAST(2.0, 1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b)))))
            END
            FROM (
                SELECT
                    COALESCE(SUM(u.x * u.y), 0.0) AS dot,
                    COALESCE(SUM(u.x * u.x), 0.0) AS norm_a,
                    COALESCE(SUM(u.y * u.y), 0.0) AS norm_b
                FROM unnest(a, b) AS u(x, y)
            ) AS stats;
        $$;
        """
    )

    embedding_col_type: sa.types.TypeEngine[object] = (
        Vector(DEFAULT_EMBEDDING_DIMENSION)
        if has_pgvector
        else EmbeddingVector(DEFAULT_EMBEDDING_DIMENSION)
    )

    op.create_table(
        "document_chunks",
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("section_title", sa.String(length=255), nullable=True),
        sa.Column("embedding", embedding_col_type, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) "
            "OR (page_start >= 1 AND page_end >= page_start)",
            name=op.f("ck_document_chunks_page_range_valid"),
        ),
        sa.CheckConstraint("char_count > 0", name=op.f("ck_document_chunks_char_count_positive")),
        sa.CheckConstraint(
            "chunk_index >= 0", name=op.f("ck_document_chunks_chunk_index_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_document_chunks_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_doc_chunk_idx"),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"], unique=False
    )
    op.create_index("ix_document_chunks_user_id", "document_chunks", ["user_id"], unique=False)

    if has_pgvector:
        op.create_index(
            "ix_document_chunks_embedding_hnsw",
            "document_chunks",
            ["embedding"],
            unique=False,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )

    op.add_column(
        "documents",
        sa.Column("processing_error_code", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("processing_error_message", sa.String(length=500), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("chunk_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_check_constraint(
        op.f("ck_documents_chunk_count_non_negative"),
        "documents",
        "chunk_count >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_documents_chunk_count_non_negative"),
        "documents",
        type_="check",
    )
    op.drop_column("documents", "chunk_count")
    op.drop_column("documents", "processed_at")
    op.drop_column("documents", "processing_error_message")
    op.drop_column("documents", "processing_error_code")

    bind = op.get_bind()
    has_hnsw = bind.execute(
        sa.text(
            "SELECT 1 FROM pg_indexes "
            "WHERE tablename = 'document_chunks' "
            "AND indexname = 'ix_document_chunks_embedding_hnsw'"
        )
    ).scalar()
    if has_hnsw:
        op.drop_index("ix_document_chunks_embedding_hnsw", table_name="document_chunks")

    op.drop_index("ix_document_chunks_user_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.execute("DROP FUNCTION IF EXISTS sg_cosine_distance(float8[], float8[])")
