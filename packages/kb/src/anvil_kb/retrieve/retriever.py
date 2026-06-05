"""Tri-mode retriever: dense / sparse / hybrid (default).

KB-M2 — Retriever now supports three modes:

dense:
    Pure vector-similarity search via the VectorStore.
    Behaviour identical to the original KB-M1 retriever.

sparse:
    Pure BM25 search via the SparseIndex (PgBM25Index or compatible).

hybrid (default):
    Both dense and sparse are called with k*4 candidates each.
    Results are fused with hand-rolled RRF (k=60) and the top-k
    fused results are returned.  No search is ever duplicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anvil_kb.retrieve.fusion import RetrievalDebug, rrf_fuse
from anvil_kb.store.base import ScoredChunk, SparseIndex, VectorStore

if TYPE_CHECKING:
    from anvil_kb.embed import Embedder
    from anvil_kb.retrieve.rerank import Reranker


class Retriever:
    """Tri-mode retriever: dense | sparse | hybrid.

    Args:
        embedder:      Embedder used to convert questions to query vectors.
        vector_store:  Dense (pgvector) store.
        sparse_index:  BM25/sparse index.  Required for 'sparse' and 'hybrid'
                       modes; raises ValueError at construction time if absent.
        mode:          One of 'dense', 'sparse', 'hybrid' (default 'hybrid').
        reranker:      Optional cross-encoder reranker.  When provided, the
                       final top-k is produced by the reranker instead of the
                       retrieval score; when None the behaviour is identical to
                       KB-M2 (zero regression).

    Raises:
        ValueError: At construction time when mode in {'sparse','hybrid'} and
                    sparse_index is None.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        *,
        sparse_index: SparseIndex | None = None,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        reranker: Reranker | None = None,
    ) -> None:
        if mode in ("sparse", "hybrid") and sparse_index is None:
            raise ValueError(
                f"mode={mode!r} requires a sparse_index but none was provided"
            )
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._mode = mode
        self._reranker = reranker

    async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
        """Return top-k chunks for *question* using the configured mode.

        hybrid path reuses a single internal search pass (no double retrieval).

        Reranker semantics:
          - reranker=None:  behaviour identical to KB-M2 (zero regression).
          - reranker set:   candidate stage runs with k*4 to give the reranker a
                            larger pool; the reranker then produces the final top-k.
            dense/sparse:  k*4 candidates → reranker.rerank(question, candidates, top=k)
            hybrid:        full RRF-fused list (all candidates, not truncated to k)
                           → reranker.rerank(question, candidates, top=k)
        """
        if self._mode == "dense":
            if self._reranker is not None:
                candidates = await self._dense_search(question, k * 4)
                return self._reranker.rerank(question, candidates, top=k)
            return await self._dense_search(question, k)

        if self._mode == "sparse":
            assert self._sparse_index is not None  # guaranteed by __init__
            if self._reranker is not None:
                candidates = await self._sparse_index.search(question, k * 4)
                return self._reranker.rerank(question, candidates, top=k)
            return await self._sparse_index.search(question, k)

        # hybrid — run once, fuse, return top-k (or reranked top-k)
        debug = await self._hybrid_debug(question, k)
        if self._reranker is not None:
            return debug.reranked or []
        return debug.fused

    async def retrieve_debug(self, question: str, k: int = 5) -> RetrievalDebug:
        """Run the full three-path pipeline and return a debug view.

        Unlike ``retrieve()``, which respects the configured mode, this method
        always executes the hybrid path (dense + sparse + RRF) regardless of
        ``self._mode``.  Because sparse search is always required, a
        ``sparse_index`` must have been provided at construction time even when
        the production mode is ``'dense'``.

        The ``reranked`` field in the returned ``RetrievalDebug`` is populated
        when a reranker is configured (None otherwise).  The ``fused`` field
        always reflects the pre-rerank RRF order for comparison.

        Raises:
            ValueError: If no ``sparse_index`` was provided (debug always needs
                        the sparse path to build the full contributions map).
        """
        if self._sparse_index is None:
            raise ValueError("retrieve_debug requires a sparse_index")
        return await self._hybrid_debug(question, k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _dense_search(self, question: str, k: int) -> list[ScoredChunk]:
        query_vector = self._embedder.embed_query(question)
        return await self._vector_store.search(query_vector, k)

    async def _hybrid_debug(self, question: str, k: int) -> RetrievalDebug:
        """Run hybrid search once and return the full debug payload.

        Candidate expansion: both dense and sparse use k*4 candidates so the
        RRF pool is larger than the final top-k, giving fusion room to work.

        Called from both ``retrieve()`` (hybrid mode, guarded by __init__) and
        ``retrieve_debug()`` (any mode, guarded by its own ValueError check), so
        ``self._sparse_index`` is always non-None here.

        The contributions map is built over the full k*4 candidate lists, so
        ranks in contributions are 1-based positions within those lists and may
        be larger than k.  This is an intentional single-pass trade-off:
        contributions are always constructed (O(k*4)) even in non-debug callers.

        Reranker handling:
        - ``fused`` always holds the pre-rerank RRF top-k (truncated to k).
        - When a reranker is configured, the *full* fused list (up to k*4
          unique chunks after RRF) is passed to the reranker so it has a
          larger pool to work with.  The reranker result is stored in
          ``reranked``; ``fused`` is unchanged (pre-rerank view).
        - When no reranker is configured, ``reranked`` is None.
        """
        # _sparse_index is guaranteed non-None by the callers' guards above
        assert self._sparse_index is not None

        candidates = k * 4

        # ── 1. Fetch candidates from both indexes ────────────────────────────
        query_vector = self._embedder.embed_query(question)
        dense_candidates = await self._vector_store.search(query_vector, candidates)
        sparse_candidates = await self._sparse_index.search(question, candidates)

        # ── 2. Fuse with RRF ─────────────────────────────────────────────────
        # full_fused: all unique chunks after fusion (up to k*4), unsorted by k
        # fused_top_k: truncated to k, used as the pre-rerank "fused" view
        full_fused = rrf_fuse([dense_candidates, sparse_candidates], k=60, top=candidates)
        fused_top_k = full_fused[:k]

        # ── 3. Build contributions map ───────────────────────────────────────
        # contributions[chunk_id_str] = {'dense': rank | None, 'sparse': rank | None}
        # Ranks are 1-based positions within the k*4 candidate lists (may be > k)
        dense_ranks: dict[str, int] = {
            str(sc.chunk.id): pos + 1
            for pos, sc in enumerate(dense_candidates)
        }
        sparse_ranks: dict[str, int] = {
            str(sc.chunk.id): pos + 1
            for pos, sc in enumerate(sparse_candidates)
        }

        all_ids = set(dense_ranks) | set(sparse_ranks)
        contributions: dict[str, dict[str, int | None]] = {
            cid: {
                "dense": dense_ranks.get(cid),
                "sparse": sparse_ranks.get(cid),
            }
            for cid in all_ids
        }

        # ── 4. Optional reranking ────────────────────────────────────────────
        # Pass the full fused list to the reranker so it has a larger pool.
        # fused_top_k is NOT affected — it always reflects the pre-rerank order.
        reranked: list[ScoredChunk] | None = None
        if self._reranker is not None:
            reranked = self._reranker.rerank(question, full_fused, top=k)

        return RetrievalDebug(
            dense=dense_candidates[:k],
            sparse=sparse_candidates[:k],
            fused=fused_top_k,
            contributions=contributions,
            reranked=reranked,
        )
