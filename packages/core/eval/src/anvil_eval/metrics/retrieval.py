"""检索指标:recall@k / precision@k。判定 = 空白归一化后的子串包含。"""


def _norm(s: str) -> str:
    return "".join(s.split())


def _hits(chunk_text: str, evidences: list[str]) -> bool:
    t = _norm(chunk_text)
    return any(_norm(e) in t for e in evidences)


def recall_at_k(retrieved_texts: list[str], evidences: list[str], k: int) -> float | None:
    if not evidences:
        return None  # 拒答用例无 evidence,召回无定义
    top = [_norm(t) for t in retrieved_texts[:k]]
    covered = sum(1 for e in evidences if any(_norm(e) in t for t in top))
    return covered / len(evidences)


def precision_at_k(retrieved_texts: list[str], evidences: list[str], k: int) -> float | None:
    if not evidences:
        return None
    top = retrieved_texts[:k]
    if not top:
        return 0.0
    return sum(1 for t in top if _hits(t, evidences)) / len(top)


def mrr(retrieved_texts: list[str], evidences: list[str]) -> float | None:
    """首个命中任一 evidence 的 chunk 排名 r(1-based)→ 1/r;全不命中→0.0;无 evidence→None。

    判定与 recall 同口径(空白归一化子串):evidence 归一化后是 chunk 归一化后的子串即命中。
    """
    if not evidences:
        return None
    for rank, text in enumerate(retrieved_texts, start=1):
        if _hits(text, evidences):
            return 1.0 / rank
    return 0.0
