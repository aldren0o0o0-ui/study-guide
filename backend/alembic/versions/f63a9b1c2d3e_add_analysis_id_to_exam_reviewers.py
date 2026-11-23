"""add_analysis_id_to_exam_reviewers

Revision ID: f63a9b1c2d3e
Revises: d88a27768762
Create Date: 2026-11-11 11:40:00.000000

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f63a9b1c2d3e'
down_revision: str | None = 'd88a27768762'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'exam_reviewers',
        sa.Column(
            'analysis_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('document_analyses.id', ondelete='SET NULL'),
            nullable=True,
        ),
    )
    op.create_index(
        'idx_exam_reviewer_analysis_id',
        'exam_reviewers',
        ['analysis_id'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('idx_exam_reviewer_analysis_id', table_name='exam_reviewers')
    op.drop_column('exam_reviewers', 'analysis_id')

