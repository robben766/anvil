"""Dense retriever — MVP entry point; future KB-M2 fusion orchestration lives here."""

from __future__ import annotations

from anvil_kb.embed import Embedder
from anvil_kb.store.base import ScoredChunk, VectorStore


class Retriever:
    """Pure-dense retriever: embed_query → vector_store.search."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def retrieve(self, question: str, k: int = 5) -> list[ScoredChunk]:
        """Embed *question* with query prefix then delegate to the vector store."""
        query_vector = self._embedder.embed_query(question)
        return await self._vector_store.search(query_vector, k)
