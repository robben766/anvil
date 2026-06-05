"""Ingest pipeline: chunk → embed (batch) → store (+ optional sparse dual-write)."""

from __future__ import annotations

import uuid

from anvil_kb.embed import Embedder
from anvil_kb.ingest.chunker import chunk_markdown
from anvil_kb.store.base import Chunk, Document, SparseIndex, VectorStore


async def ingest_markdown(
    *,
    title: str,
    source_name: str,
    text: str,
    embedder: Embedder,
    store: VectorStore,
    size: int = 600,
    overlap: int = 100,
    sparse_index: SparseIndex | None = None,
) -> tuple[Document, int]:
    """Chunk, embed (single batch call), and store a markdown document.

    Args:
        title:        Human-readable document title.
        source_name:  Stable logical identifier (used for idempotent re-ingest).
        text:         Raw markdown content.
        embedder:     Embedder instance (embed_texts called once for the batch).
        store:        Vector store — receives upsert_chunks.
        size:         Chunk character target size (default 600).
        overlap:      Overlap between consecutive chunks (default 100).
        sparse_index: Optional BM25/sparse index for dual-write.  When provided,
                      ``sparse_index.index_chunks(chunks)`` is called *after*
                      ``store.upsert_chunks`` so that any cascaded deletion of
                      old postings (triggered by the upsert) happens first.

    Returns:
        (Document, chunk_count). Raises ValueError if text produces no chunks.
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

    # ── 1. Vector store (always) ─────────────────────────────────────────────
    await store.upsert_chunks(doc, chunks)

    # ── 2. Sparse dual-write (optional) ─────────────────────────────────────
    # Order matters: upsert first so cascaded deletion of old postings completes
    # before we re-index.  PgBM25Index.index_chunks is idempotent (deletes old
    # postings for these chunk ids before inserting new ones).
    # 不一致窗口: index_chunks 失败时向量已提交而 postings 缺失;
    # 重灌同文档即可自愈 (upsert 级联清旧)
    if sparse_index is not None:
        await sparse_index.index_chunks(chunks)

    return doc, len(chunks)
