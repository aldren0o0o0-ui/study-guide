"""remove_folders

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-11-14 15:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_documents_folder_id", table_name="documents")
    op.drop_constraint(op.f("fk_documents_folder_id_folders"), "documents", type_="foreignkey")
    op.drop_column("documents", "folder_id")
    op.drop_index("ix_folders_user_id", table_name="folders")
    op.drop_table("folders")


def downgrade() -> None:
    op.create_table(
        "folders",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_folders_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_folders")),
        sa.UniqueConstraint("user_id", "name", name="uq_folders_user_id_name"),
    )
    op.create_index("ix_folders_user_id", "folders", ["user_id"], unique=False)
    op.add_column("documents", sa.Column("folder_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        op.f("fk_documents_folder_id_folders"),
        "documents",
        "folders",
        ["folder_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_documents_folder_id", "documents", ["folder_id"], unique=False)

