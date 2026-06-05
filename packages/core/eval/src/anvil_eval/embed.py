"""本地 embedding(fastembed/ONNX,CPU)。百炼后置期间的无云依赖方案。"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_model = None


def embed(texts: list[str]) -> np.ndarray:
    """返回 shape (n, dim) 的 L2 归一化向量。首次调用会下载模型(~100MB)。"""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=_MODEL_NAME)
    vecs = np.array(list(_model.embed(texts)))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)
