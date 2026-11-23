"""create_document_assets

Revision ID: d88a27768762
Revises: 571b0ba37d5e
Create Date: 2026-11-10 21:27:24.852346

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd88a27768762'
down_revision: str | None = '571b0ba37d5e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'document_assets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('storage_key', sa.String(length=512), nullable=False),
        sa.Column('asset_type', sa.String(length=32), server_default='image', nullable=False),
        sa.Column('mime_type', sa.String(length=128), nullable=False),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('checksum_sha256', sa.String(length=64), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('section_title', sa.String(length=255), nullable=True),
        sa.Column('caption', sa.String(length=500), nullable=True),
        sa.Column('order_index', sa.Integer(), server_default='0', nullable=False),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('file_size_bytes > 0', name='asset_file_size_positive'),
        sa.CheckConstraint('order_index >= 0', name='asset_order_index_non_negative'),
    )
    op.create_index('ix_document_assets_document_id', 'document_assets', ['document_id'])
    op.create_index('ix_document_assets_user_id', 'document_assets', ['user_id'])
    op.create_index('ix_document_assets_storage_key', 'document_assets', ['storage_key'], unique=True)
    op.create_index('ix_document_assets_doc_checksum', 'document_assets', ['document_id', 'checksum_sha256'])


def downgrade() -> None:
    op.drop_index('ix_document_assets_doc_checksum', table_name='document_assets')
    op.drop_index('ix_document_assets_storage_key', table_name='document_assets')
    op.drop_index('ix_document_assets_user_id', table_name='document_assets')
    op.drop_index('ix_document_assets_document_id', table_name='document_assets')
    op.drop_table('document_assets')
