"""cross-encoder 重排:对 (question, chunk_content) 逐对打分,精排候选。

双塔(bi-encoder)检索快但只看向量近似;交叉编码(cross-encoder)把 query 和文档
拼一起过模型,精度高但每对都要前向一次——只能用在小候选集的最后一级。

TextCrossEncoder API 实探结论(fastembed 版本确认):
  - 签名: rerank(self, query: str, documents: Iterable[str], ...) -> Iterable[float]
  - 返回生成器,按输入顺序逐一 yield 分数(位置 i 的分数对应 documents[i])
  - 不做任何排序,调用方负责 zip + sort
  - 使用时需 list() 完整消费生成器,否则只迭代一次
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from anvil_kb.store.base import ScoredChunk

_MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker(Protocol):
    def rerank(
        self, question: str, candidates: list[ScoredChunk], *, top: int
    ) -> list[ScoredChunk]: ...


class FastEmbedReranker:
    """Cross-encoder reranker using fastembed TextCrossEncoder (bge-reranker-base).

    懒加载:构造时不下载模型,首次调用 rerank() 时才载入(~1 GB)。
    """

    def __init__(self) -> None:
        self._model = None  # 懒加载(模型 ~1GB,首次调用才下载/载入)

    def _ensure(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(_MODEL_NAME)
        return self._model

    def rerank(
        self,
        question: str,
        candidates: list[ScoredChunk],
        *,
        top: int,
    ) -> list[ScoredChunk]:
        """Rerank candidates by cross-encoder score, return top-k in descending order.

        Args:
            question:   The user query.
            candidates: Candidate chunks to rerank (may be k*4 or fewer).
            top:        Maximum number of results to return.

        Returns:
            Up to *top* ScoredChunk objects, sorted descending by cross-encoder score.
            The .score field is replaced with the raw cross-encoder logit/score.
        """
        from anvil_kb.store.base import ScoredChunk as _ScoredChunk

        if not candidates:
            return []

        # TextCrossEncoder.rerank returns a generator of floats, one per input
        # document, in the same order as the input (no internal sorting).
        # We must list() it to fully consume the generator before zip.
        scores = list(
            self._ensure().rerank(question, [c.chunk.content for c in candidates])
        )

        # Zip candidates with their scores, sort descending, take top-k
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            _ScoredChunk(chunk=c.chunk, score=float(s)) for c, s in ranked[:top]
        ]
