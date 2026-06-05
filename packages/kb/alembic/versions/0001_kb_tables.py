"""kb tables: documents + chunks with pgvector

Revision ID: 0001_kb_tables
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

revision = "0001_kb_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "kb_documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("source_name", sa.Text, nullable=False, unique=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_table(
        "kb_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("kb_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("header_path", sa.Text, nullable=False, server_default=""),
        sa.Column("start_offset", sa.Integer, nullable=False),
        sa.Column("end_offset", sa.Integer, nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_kb_chunks_document_id", "kb_chunks", ["document_id"])
    op.execute(
        "CREATE INDEX ix_kb_chunks_embedding ON kb_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_embedding")
    op.drop_index("ix_kb_chunks_document_id", table_name="kb_chunks")
    op.drop_table("kb_chunks")
    op.drop_table("kb_documents")
