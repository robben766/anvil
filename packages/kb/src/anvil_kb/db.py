"""kb 数据模型:kb_documents / kb_chunks(pgvector)。"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

EMBEDDING_DIM = 512  # BAAI/bge-small-zh-v1.5


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "kb_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ChunkRow(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("kb_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    header_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


_DEFAULT_DB_URL = "postgresql+asyncpg://anvil:anvil@localhost:5432/anvil"


def make_engine(database_url: str | None = None) -> AsyncEngine:
    """Create an async engine with NullPool (mirrors ledger.py pattern).

    Priority: explicit argument > ANVIL_DATABASE_URL env var > built-in default.
    """
    url = database_url or os.environ.get("ANVIL_DATABASE_URL", _DEFAULT_DB_URL)
    return create_async_engine(url, poolclass=NullPool)


def make_session_factory(
    database_url: str | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Return an async_sessionmaker bound to an engine built from *database_url*."""
    engine = make_engine(database_url)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
