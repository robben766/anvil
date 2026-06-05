"""上下文召回 = reference 的主张中能在 contexts 找到支持的比例(检索漏没漏)。"""

from __future__ import annotations

from anvil_eval.metrics.claims import split_claims, supported_ratio


async def context_recall(*, reference: str, contexts: list[str]) -> float:
    claims = await split_claims(reference)
    return await supported_ratio(claims, contexts)
