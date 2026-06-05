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
    enrich_chat=None,
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
        enrich_chat:  Optional chat function for Contextual Retrieval enrichment.
                      When non-None, ``enrich_chunks`` is called to generate a
                      context_prefix for each chunk; the prefix is prepended to
                      the embedding input (prefix + "\\n" + content).  None means
                      no enrichment (zero behaviour change, prefix stays "").

    Returns:
        (Document, chunk_count). Raises ValueError if text produces no chunks.
    """
    if not text.strip():
        raise ValueError("document produced no chunks")

    drafts = chunk_markdown(text, size=size, overlap=overlap)
    if not drafts:
        raise ValueError("document produced no chunks")

    # ── Optional: Contextual Retrieval enrichment ────────────────────────────
    # Must happen BEFORE embedding so that prefixed text is used for vectors.
    contexts: list[str] = [""] * len(drafts)
    if enrich_chat is not None:
        from anvil_kb.ingest.enrich import enrich_chunks  # noqa: PLC0415

        contexts = await enrich_chunks(
            doc_text=text, drafts=drafts, chat=enrich_chat
        )

    # ── Single batch embed call — never per-chunk ────────────────────────────
    # Embedding input: prefix + "\n" + content when prefix present, else content.
    embed_inputs = [
        (contexts[i] + "\n" + d.content if contexts[i] else d.content)
        for i, d in enumerate(drafts)
    ]
    vectors = embedder.embed_texts(embed_inputs)

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
            context_prefix=contexts[i],
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
