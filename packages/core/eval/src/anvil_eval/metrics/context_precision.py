"""上下文精度(ranking-aware):相关项排得越靠前分越高。"""

from __future__ import annotations

from anvil_eval.judge import judge_json

_RUBRIC = (
    "判断该 context 片段对回答问题是否有用(直接相关才算)。"
    "输出 JSON: {\"reason\": str, \"relevant\": bool}。"
)


def precision_from_flags(flags: list[bool]) -> float:
    relevant_total = sum(flags)
    if relevant_total == 0:
        return 0.0
    score = 0.0
    hit = 0
    for k, flag in enumerate(flags, 1):
        if flag:
            hit += 1
            score += hit / k
    return score / relevant_total


async def context_precision(*, question: str, contexts: list[str]) -> float:
    flags: list[bool] = []
    for ctx in contexts:
        out = await judge_json(_RUBRIC, {"question": question, "context": ctx})
        flags.append(bool(out.get("relevant")))
    return precision_from_flags(flags)
