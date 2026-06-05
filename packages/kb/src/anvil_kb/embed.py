"""本地 embedding:协议 + fastembed bge-small-zh-v1.5(512 维)。

bge 中文系列约定:query 侧加指令前缀,passage 侧不加。
"""

from __future__ import annotations

from typing import Protocol

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章:"


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self) -> None:
        self._model = None  # 懒加载:首次调用才下载/载入模型

    @property
    def dim(self) -> int:
        return 512

    def _ensure(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=_MODEL_NAME)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._ensure().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([_QUERY_PREFIX + text])[0]
