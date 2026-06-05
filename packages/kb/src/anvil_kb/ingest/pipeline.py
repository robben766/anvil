"""Ingest pipeline: chunk → embed (batch) → store."""

from __future__ import annotations

import uuid

from anvil_kb.embed import Embedder
from anvil_kb.ingest.chunker import chunk_markdown
from anvil_kb.store.base import Chunk, Document, VectorStore


async def ingest_markdown(
    *,
    title: str,
    source_name: str,
    text: str,
    embedder: Embedder,
    store: VectorStore,
    size: int = 600,
    overlap: int = 100,
) -> tuple[Document, int]:
    """Chunk, embed (single batch call), and store a markdown document.

    Returns (Document, chunk_count). Raises ValueError if text produces no chunks.
    """
    if not text.strip():
        raise ValueError("document produced no chunks")

    drafts = chunk_markdown(text, size=size, overlap=overlap)
    if not drafts:
        raise ValueError("document produced no chunks")

    # Single batch embed call — never per-chunk
    vectors = embedder.embed_texts([d.content for d in drafts])

    doc = Document(
        id=uuid.uuid4(),
        title=title,
        source_name=source_name,
        content=text,
    )

    chunks = [
        Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            seq=draft.seq,
            content=draft.content,
            header_path=draft.header_path,
            start_offset=draft.start_offset,
            end_offset=draft.end_offset,
            embedding=vectors[i],
        )
        for i, draft in enumerate(drafts)
    ]

    await store.upsert_chunks(doc, chunks)
    return doc, len(chunks)
