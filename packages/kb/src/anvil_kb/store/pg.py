"""PgVectorStore: async SQLAlchemy + pgvector cosine similarity store."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_kb.db import ChunkRow, DocumentRow
from anvil_kb.store.base import Chunk, Document, ScoredChunk


class PgVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert_chunks(self, doc: Document, chunks: list[Chunk]) -> None:
        """Insert doc + chunks in a single transaction; replace same source_name."""
        async with self._session_factory() as session:
            async with session.begin():
                # Delete existing doc with same source_name (CASCADE removes its chunks)
                await session.execute(
                    delete(DocumentRow).where(DocumentRow.source_name == doc.source_name)
                )

                # Insert new document row and flush so FK is visible for chunks
                session.add(
                    DocumentRow(
                        id=doc.id,
                        title=doc.title,
                        source_name=doc.source_name,
                        content=doc.content,
                    )
                )
                await session.flush()

                # Insert chunk rows
                for chunk in chunks:
                    session.add(
                        ChunkRow(
                            id=chunk.id,
                            document_id=chunk.document_id,
                            seq=chunk.seq,
                            content=chunk.content,
                            header_path=chunk.header_path,
                            start_offset=chunk.start_offset,
                            end_offset=chunk.end_offset,
                            embedding=chunk.embedding,
                            context_prefix=chunk.context_prefix,
                        )
                    )

    async def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        """Return top-k chunks ordered by cosine similarity (score = 1 - distance)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    ChunkRow.id,
                    ChunkRow.document_id,
                    ChunkRow.seq,
                    ChunkRow.content,
                    ChunkRow.header_path,
                    ChunkRow.start_offset,
                    ChunkRow.end_offset,
                    ChunkRow.context_prefix,
                    ChunkRow.embedding.cosine_distance(query_vector).label("distance"),
                )
                .order_by("distance")
                .limit(k)
            )
            rows = result.all()

        return [
            ScoredChunk(
                chunk=Chunk(
                    id=row.id,
                    document_id=row.document_id,
                    seq=row.seq,
                    content=row.content,
                    header_path=row.header_path,
                    start_offset=row.start_offset,
                    end_offset=row.end_offset,
                    embedding=None,  # 省流量,不带回 embedding
                    context_prefix=row.context_prefix,
                ),
                score=1.0 - float(row.distance),
            )
            for row in rows
        ]

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete document by id; silently succeeds if not found."""
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(DocumentRow).where(DocumentRow.id == document_id)
                )
