"""Unit tests for Retriever — fully mocked, no real embedder or vector store."""

from __future__ import annotations

import uuid

import pytest

from anvil_kb.retrieve.retriever import Retriever
from anvil_kb.store.base import Chunk, ScoredChunk

# Sentinel to distinguish dense vs sparse calls
_DENSE_SENTINEL = "dense_result"
_SPARSE_SENTINEL = "sparse_result"

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
    retriever = Retriever(FakeEmbedder(), store, mode="dense")

    await retriever.retrieve("等待期?", k=1)

    assert store.last_vector == _SENTINEL_QUERY_VEC, (
        "search() should receive embed_query vector, not embed_texts vector"
    )
    assert store.last_vector != _SENTINEL_TEXT_VEC


@pytest.mark.asyncio
async def test_retrieve_passes_k_to_store():
    """k parameter must be forwarded verbatim to vector_store.search."""
    store = FakeVectorStore([])
    retriever = Retriever(FakeEmbedder(), store, mode="dense")

    await retriever.retrieve("question", k=7)

    assert store.last_k == 7


@pytest.mark.asyncio
async def test_retrieve_returns_store_results_unchanged():
    """retrieve() should return exactly what vector_store.search returns."""
    chunks = [_scored("chunk A", seq=0), _scored("chunk B", seq=1)]
    store = FakeVectorStore(chunks)
    retriever = Retriever(FakeEmbedder(), store, mode="dense")

    result = await retriever.retrieve("q", k=2)

    assert result == chunks


@pytest.mark.asyncio
async def test_retrieve_default_k_is_5():
    """Default k should be 5 (dense mode test)."""
    store = FakeVectorStore([])
    retriever = Retriever(FakeEmbedder(), store, mode="dense")

    await retriever.retrieve("q")

    assert store.last_k == 5


# ---------------------------------------------------------------------------
# Fake sparse index for mode tests
# ---------------------------------------------------------------------------


class FakeSparseIndex:
    """Sparse index that returns sentinel chunks to distinguish from dense."""

    def __init__(self, results: list[ScoredChunk] | None = None) -> None:
        self._results: list[ScoredChunk] = results or []
        self.last_query: str | None = None
        self.last_k: int | None = None

    async def index_chunks(self, chunks):
        pass

    async def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.last_query = query
        self.last_k = k
        return self._results[:k]


def _make_distinct_chunks(
    prefix: str, count: int
) -> list[ScoredChunk]:
    """Create *count* ScoredChunk objects with content prefixed by *prefix*."""
    return [
        ScoredChunk(
            chunk=_make_chunk(f"{prefix}_{i}", seq=i),
            score=1.0 - 0.05 * i,
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Mode routing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_dense_uses_vector_store_only():
    """mode='dense' must call vector_store.search and NOT sparse.search."""
    dense_results = _make_distinct_chunks("dense", 3)
    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(_make_distinct_chunks("sparse", 3))

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse, mode="dense")
    result = await retriever.retrieve("q", k=3)

    assert store.last_k == 3, "vector store must be called with k=3"
    assert sparse.last_k is None, "sparse.search must NOT be called in dense mode"
    # Results come from dense store only
    result_ids = {r.chunk.id for r in result}
    dense_ids = {r.chunk.id for r in dense_results}
    assert result_ids <= dense_ids


@pytest.mark.asyncio
async def test_mode_sparse_uses_sparse_index_only():
    """mode='sparse' must call sparse.search and NOT vector_store.search."""
    dense_results = _make_distinct_chunks("dense", 3)
    sparse_results = _make_distinct_chunks("sparse", 3)
    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse, mode="sparse")
    result = await retriever.retrieve("q", k=3)

    assert sparse.last_k == 3, "sparse.search must be called with k=3"
    assert store.last_k is None, "vector_store.search must NOT be called in sparse mode"
    result_ids = {r.chunk.id for r in result}
    sparse_ids = {r.chunk.id for r in sparse_results}
    assert result_ids <= sparse_ids


@pytest.mark.asyncio
async def test_mode_hybrid_calls_both_with_k_times_4():
    """mode='hybrid' must call both stores with k*4 candidates."""
    dense_results = _make_distinct_chunks("dense", 20)
    sparse_results = _make_distinct_chunks("sparse", 20)
    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse, mode="hybrid")
    result = await retriever.retrieve("q", k=5)

    assert store.last_k == 20, f"dense k must be k*4=20, got {store.last_k}"
    assert sparse.last_k == 20, f"sparse k must be k*4=20, got {sparse.last_k}"
    assert len(result) == 5, f"hybrid result must be truncated to k=5, got {len(result)}"


@pytest.mark.asyncio
async def test_mode_hybrid_is_default():
    """Default mode is 'hybrid'."""
    store = FakeVectorStore(_make_distinct_chunks("dense", 5))
    sparse = FakeSparseIndex(_make_distinct_chunks("sparse", 5))

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse)
    await retriever.retrieve("q", k=2)

    # Both must be called (hybrid behaviour)
    assert store.last_k is not None
    assert sparse.last_k is not None


# ---------------------------------------------------------------------------
# ValueError when no sparse_index but mode requires it
# ---------------------------------------------------------------------------


def test_sparse_mode_without_sparse_index_raises():
    """Constructing Retriever(mode='sparse') without sparse_index → ValueError."""
    with pytest.raises(ValueError):
        Retriever(FakeEmbedder(), FakeVectorStore([]), mode="sparse")


def test_hybrid_mode_without_sparse_index_raises():
    """Constructing Retriever(mode='hybrid') without sparse_index → ValueError."""
    with pytest.raises(ValueError):
        Retriever(FakeEmbedder(), FakeVectorStore([]), mode="hybrid")


def test_dense_mode_without_sparse_index_is_ok():
    """Constructing Retriever(mode='dense') without sparse_index is valid."""
    retriever = Retriever(FakeEmbedder(), FakeVectorStore([]), mode="dense")
    assert retriever is not None


@pytest.mark.asyncio
async def test_retrieve_debug_without_sparse_raises():
    """retrieve_debug() must raise ValueError when no sparse_index is available.

    Even a Retriever constructed in 'dense' mode (which normally needs no
    sparse_index) must raise because retrieve_debug always runs the hybrid
    path and requires both indexes.
    """
    retriever = Retriever(FakeEmbedder(), FakeVectorStore([]), mode="dense")
    with pytest.raises(ValueError, match="sparse_index"):
        await retriever.retrieve_debug("any question")


# ---------------------------------------------------------------------------
# retrieve_debug contributions test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_debug_contributions_ranks():
    """contributions must include correct ranks for intersection and unique items.

    Setup:
      - shared chunk appears in both dense (rank 1) and sparse (rank 1) results
      - dense_only appears only in dense (rank 2)
      - sparse_only appears only in sparse (rank 2)

    Expected contributions:
      shared:     {'dense': 1, 'sparse': 1}
      dense_only: {'dense': 2, 'sparse': None}
      sparse_only: {'dense': None, 'sparse': 2}
    """
    shared = _make_chunk("shared", seq=0)
    dense_only = _make_chunk("dense_only", seq=1)
    sparse_only = _make_chunk("sparse_only", seq=2)

    dense_results = [
        ScoredChunk(chunk=shared, score=0.9),
        ScoredChunk(chunk=dense_only, score=0.8),
    ]
    sparse_results = [
        ScoredChunk(chunk=shared, score=0.85),
        ScoredChunk(chunk=sparse_only, score=0.75),
    ]

    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse, mode="hybrid")
    debug = await retriever.retrieve_debug("q", k=5)

    contributions = debug.contributions
    shared_str = str(shared.id)
    dense_only_str = str(dense_only.id)
    sparse_only_str = str(sparse_only.id)

    assert contributions[shared_str]["dense"] == 1
    assert contributions[shared_str]["sparse"] == 1

    assert contributions[dense_only_str]["dense"] == 2
    assert contributions[dense_only_str]["sparse"] is None

    assert contributions[sparse_only_str]["dense"] is None
    assert contributions[sparse_only_str]["sparse"] == 2


@pytest.mark.asyncio
async def test_retrieve_debug_returns_correct_k_lengths():
    """retrieve_debug.dense/sparse/fused are all truncated to k items."""
    dense_results = _make_distinct_chunks("dense", 20)
    sparse_results = _make_distinct_chunks("sparse", 20)
    store = FakeVectorStore(dense_results)
    sparse = FakeSparseIndex(sparse_results)

    retriever = Retriever(FakeEmbedder(), store, sparse_index=sparse, mode="hybrid")
    debug = await retriever.retrieve_debug("q", k=3)

    assert len(debug.dense) == 3
    assert len(debug.sparse) == 3
    assert len(debug.fused) == 3


@pytest.mark.asyncio
async def test_retrieve_hybrid_no_double_search():
    """retrieve() in hybrid mode must not call stores twice.
    Internal implementation must reuse one search pass.
    """
    call_counts: dict[str, int] = {"dense": 0, "sparse": 0}

    class CountingStore:
        last_k = None

        async def upsert_chunks(self, doc, chunks):
            pass

        async def search(self, query_vector, k):
            call_counts["dense"] += 1
            self.last_k = k
            return _make_distinct_chunks("dense", k)

        async def delete_document(self, document_id):
            pass

    class CountingSparse:
        last_k = None

        async def index_chunks(self, chunks):
            pass

        async def search(self, query, k):
            call_counts["sparse"] += 1
            self.last_k = k
            return _make_distinct_chunks("sparse", k)

    counting_store = CountingStore()
    counting_sparse = CountingSparse()
    retriever = Retriever(
        FakeEmbedder(), counting_store, sparse_index=counting_sparse, mode="hybrid"
    )
    await retriever.retrieve("q", k=5)

    assert call_counts["dense"] == 1, "dense search called more than once"
    assert call_counts["sparse"] == 1, "sparse search called more than once"
