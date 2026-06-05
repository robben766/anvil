"""忠实度 = answer 的主张中被 contexts 支持的比例(幻觉的反面)。"""

from __future__ import annotations

from anvil_eval.metrics.claims import split_claims, supported_ratio


async def faithfulness(*, answer: str, contexts: list[str]) -> float:
    claims = await split_claims(answer)
    return await supported_ratio(claims, contexts)
