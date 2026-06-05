from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Document:
    id: uuid.UUID
    title: str
    source_name: str
    content: str


@dataclass
class Chunk:
    id: uuid.UUID
    document_id: uuid.UUID
    seq: int
    content: str
    header_path: str
    start_offset: int
    end_offset: int
    embedding: list[float] | None = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # 统一语义:越大越相关


class VectorStore(Protocol):
    async def upsert_chunks(self, doc: Document, chunks: list[Chunk]) -> None: ...
    async def search(self, query_vector: list[float], k: int) -> list[ScoredChunk]: ...
    async def delete_document(self, document_id: uuid.UUID) -> None: ...


class SparseIndex(Protocol):  # KB-M2 实现;Day-1 只定义接口
    async def index_chunks(self, chunks: list[Chunk]) -> None: ...
    async def search(self, query: str, k: int) -> list[ScoredChunk]: ...
