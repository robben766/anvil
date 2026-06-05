"""Unit tests for Retriever — fully mocked, no real embedder or vector store."""

from __future__ import annotations

import uuid

import pytest

from anvil_kb.retrieve.retriever import Retriever
from anvil_kb.store.base import Chunk, ScoredChunk

# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------

_SENTINEL_QUERY_VEC = [0.1, 0.2, 0.3]
_SENTINEL_TEXT_VEC = [0.9, 0.8, 0.7]


class FakeEmbedder:
    """Returns sentinel vectors that are distinguishable from embed_texts."""

    @property
    def dim(self) -> int:
        return 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_SENTINEL_TEXT_VEC for _ in texts]

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return _SENTINEL_QUERY_VEC


def _make_chunk(content: str, seq: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        seq=seq,
        content=content,
        header_path="",
        start_offset=0,
        end_offset=len(content),
    )


def _scored(content: str, seq: int = 0) -> ScoredChunk:
    return ScoredChunk(chunk=_make_chunk(content, seq), score=0.9 - seq * 0.1)


class FakeVectorStore:
    """Records calls so tests can assert on them."""

    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results
        self.last_vector: list[float] | None = None
        self.last_k: int | None = None

    async def upsert_chunks(self, doc, chunks):  # type: ignore[override]
        pass

    async def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]:
        self.last_vector = query_vector
        self.last_k = k
        return self._results[:k]

    async def delete_document(self, document_id):  # type: ignore[override]
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_uses_embed_query_vector():
    """retrieve() must call embed_query (not embed_texts) and pass that vector to search."""
    expected = [_scored("等待期为90天。", seq=0)]
    store = FakeVectorStore(expected)
    retriever = Retriever(FakeEmbedder(), store)

    await retriever.retrieve("等待期?", k=1)

    assert store.last_vector == _SENTINEL_QUERY_VEC, (
        "search() should receive embed_query vector, not embed_texts vector"
    )
    assert store.last_vector != _SENTINEL_TEXT_VEC


@pytest.mark.asyncio
async def test_retrieve_passes_k_to_store():
    """k parameter must be forwarded verbatim to vector_store.search."""
    store = FakeVectorStore([])
    retriever = Retriever(FakeEmbedder(), store)

    await retriever.retrieve("question", k=7)

    assert store.last_k == 7


@pytest.mark.asyncio
async def test_retrieve_returns_store_results_unchanged():
    """retrieve() should return exactly what vector_store.search returns."""
    chunks = [_scored("chunk A", seq=0), _scored("chunk B", seq=1)]
    store = FakeVectorStore(chunks)
    retriever = Retriever(FakeEmbedder(), store)

    result = await retriever.retrieve("q", k=2)

    assert result == chunks


@pytest.mark.asyncio
async def test_retrieve_default_k_is_5():
    """Default k should be 5."""
    store = FakeVectorStore([])
    retriever = Retriever(FakeEmbedder(), store)

    await retriever.retrieve("q")

    assert store.last_k == 5
