"""答案相关性:从 answer 反推 N 个问题,与原问题的 embedding 余弦均值。"""

from __future__ import annotations

import numpy as np

from anvil_eval.embed import embed
from anvil_eval.judge import judge_json

_RUBRIC = (
    "根据给定答案,反推出 3 个该答案最可能在回答的问题。"
    "输出 JSON: {\"reason\": str, \"questions\": [str, str, str]}。"
)


def _mean_cosine(q: np.ndarray, others: np.ndarray) -> float:
    return float(np.mean(others @ q))


async def answer_relevancy(*, question: str, answer: str) -> float:
    out = await judge_json(_RUBRIC, {"answer": answer})
    reversed_qs = list(out.get("questions") or [])
    if not reversed_qs:
        return 0.0
    vecs = embed([question, *reversed_qs])
    return _mean_cosine(vecs[0], vecs[1:])
