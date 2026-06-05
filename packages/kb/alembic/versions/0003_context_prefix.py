"""Add context_prefix column to kb_chunks

Revision ID: 0003_context_prefix
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_context_prefix"
down_revision = "0002_bm25_postings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "kb_chunks",
        sa.Column(
            "context_prefix",
            sa.Text,
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    op.drop_column("kb_chunks", "context_prefix")
