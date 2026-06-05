"""主张拆分与逐条比对 —— faithfulness 与 context_recall 的共用骨架。"""

from __future__ import annotations

from anvil_eval.judge import judge_json

_SPLIT_RUBRIC = (
    "把给定文本拆分成独立的原子主张(每条只含一个可验证的事实)。"
    "输出 JSON: {\"reason\": str, \"claims\": [str, ...]}。无主张则 claims 为空数组。"
)

_VERIFY_RUBRIC = (
    "判断主张是否能由给定 context 支持(明确蕴含才算支持,常识推断不算)。"
    "输出 JSON: {\"reason\": str, \"supported\": bool}。"
)


async def split_claims(text: str) -> list[str]:
    out = await judge_json(_SPLIT_RUBRIC + "(任务:拆分)", {"text": text})
    return list(out.get("claims") or [])


async def supported_ratio(claims: list[str], contexts: list[str]) -> float:
    if not claims:
        return 0.0
    ctx = "\n".join(contexts)
    supported = 0
    for claim in claims:
        out = await judge_json(_VERIFY_RUBRIC, {"claim": claim, "context": ctx})
        if out.get("supported"):
            supported += 1
    return supported / len(claims)
