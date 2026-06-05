"""BM25 postings table + token_count column on kb_chunks

Revision ID: 0002_bm25_postings
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0002_bm25_postings"
down_revision = "0001_kb_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add token_count to kb_chunks (default 0 for existing rows)
    op.add_column(
        "kb_chunks",
        sa.Column(
            "token_count",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
    )

    # Create kb_postings with composite PK (term, chunk_id)
    op.create_table(
        "kb_postings",
        sa.Column("term", sa.Text, nullable=False),
        sa.Column(
            "chunk_id",
            UUID(as_uuid=True),
            sa.ForeignKey("kb_chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tf", sa.Integer, nullable=False),
        sa.PrimaryKeyConstraint("term", "chunk_id"),
    )
    op.create_index("ix_kb_postings_term", "kb_postings", ["term"])


def downgrade() -> None:
    op.drop_index("ix_kb_postings_term", table_name="kb_postings")
    op.drop_table("kb_postings")
    op.drop_column("kb_chunks", "token_count")
