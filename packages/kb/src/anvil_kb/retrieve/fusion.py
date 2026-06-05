"""Reciprocal Rank Fusion (RRF) — hand-rolled, no third-party library.

RRF formula (Cormack, Clarke, Buettcher 2009):
    score(c) = Σ_lists  1 / (k + rank_i(c))

where rank_i(c) is the 1-based position of chunk c in list i (lists that do
not contain c contribute 0 to the sum).  k=60 is the original paper's value
and smooths the impact of very high-ranked items.

Tie-breaking rule:
    When two chunks have identical fused scores, the one whose *first
    occurrence* (in list-index order, then in within-list rank order) appears
    earliest comes first.  This makes the sort fully deterministic regardless
    of UUID ordering.

Chunk identity rule:
    If the same chunk.id appears in more than one list, the ScoredChunk
    object from the *first* occurrence (earliest list index, earliest rank in
    that list) is used in the output; only the .score field is replaced with
    the fused value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anvil_kb.store.base import ScoredChunk

if TYPE_CHECKING:
    pass


@dataclass
class RetrievalDebug:
    """Debug view of a hybrid retrieval.

    Attributes:
        dense:         Top-k results from the dense (vector) path.
        sparse:        Top-k results from the sparse (BM25) path.
        fused:         Final top-k results after RRF fusion.
        contributions: For each chunk_id (str), maps "dense" and/or "sparse"
                       to the 1-based rank within the *k*4 candidate list*
                       used internally (None if the chunk was absent from that
                       sub-list).  Because the candidate list has k*4 entries,
                       ranks may be greater than k — e.g. rank 17 in a k=5
                       search (k*4 = 20 candidates).
    """

    dense: list[ScoredChunk]
    sparse: list[ScoredChunk]
    fused: list[ScoredChunk]
    contributions: dict[str, dict[str, int | None]]


def rrf_fuse(
    ranked_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    top: int = 5,
) -> list[ScoredChunk]:
    """Fuse multiple ranked lists into one via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is a pre-ranked sequence of ScoredChunk
                      objects (index 0 = best, i.e. rank 1).  May be empty.
        k:            Smoothing constant (default 60, per original RRF paper).
        top:          Maximum number of results to return.

    Returns:
        A new list of ScoredChunk with .score replaced by the RRF fused score,
        sorted descending by score, truncated to *top* items.

    Algorithm:
        1. Iterate every list; for each item at 0-based position pos, its
           1-based rank = pos + 1 and its RRF contribution = 1 / (k + rank).
        2. Accumulate contributions keyed by chunk.id (UUID).
        3. For each unique chunk.id record:
               - cumulative fused score
               - first-occurrence coordinates (list_idx, rank)  ← for tie-break
               - ScoredChunk object from the first occurrence   ← for output
        4. Sort descending by fused_score; tie-break ascending by
           (first_list_idx, first_rank) so earlier-seen items win.
        5. Slice to *top* and return with .score overwritten.
    """
    if not ranked_lists:
        return []

    # chunk_id → (fused_score, first_list_idx, first_rank, ScoredChunk object)
    accumulator: dict[
        object,  # uuid.UUID used as key
        tuple[float, int, int, ScoredChunk],
    ] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        for pos, sc in enumerate(ranked):
            rank = pos + 1  # 1-based
            contribution = 1.0 / (k + rank)
            cid = sc.chunk.id

            if cid in accumulator:
                old_score, first_li, first_rank, first_sc = accumulator[cid]
                accumulator[cid] = (old_score + contribution, first_li, first_rank, first_sc)
            else:
                # First time we see this chunk — record coordinates for tie-break
                accumulator[cid] = (contribution, list_idx, rank, sc)

    if not accumulator:
        return []

    # Sort: primary = fused_score DESC; tie-break = (first_list_idx, first_rank) ASC
    sorted_items = sorted(
        accumulator.values(),
        key=lambda t: (-t[0], t[1], t[2]),
    )

    result: list[ScoredChunk] = []
    for fused_score, _li, _rank, sc in sorted_items[:top]:
        # Replace score with the fused value; keep the original Chunk object
        result.append(ScoredChunk(chunk=sc.chunk, score=fused_score))

    return result
