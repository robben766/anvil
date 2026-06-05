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
