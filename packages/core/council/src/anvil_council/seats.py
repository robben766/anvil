"""Jurors score a case independently and in parallel via guard.structured_chat.

Each juror is a model. Jurors do NOT see each other's scores — evaluation wants
independence (this also avoids sycophancy/anchoring). The prompt mentions JSON so
DeepSeek's json_object mode engages."""

from __future__ import annotations

import asyncio
from typing import Any

from anvil_guard import structured_chat

from anvil_council.rubric import DEFAULT_RUBRIC, Rubric
from anvil_council.verdict import JurorScore

_SYSTEM = (
    "你是严格的评测陪审员。仅依据给定资料,对候选答案按每个维度打 0~1 分"
    "(1=完全满足,0=完全不满足)。先写 reason 再给分,只输出一个 JSON 对象。"
)


def _build_messages(case: dict[str, Any], rubric: Rubric) -> list[dict[str, str]]:
    crit_lines = "\n".join(f"- {c.key}: {c.description}" for c in rubric.criteria)
    keys = ", ".join(rubric.keys())
    user = (
        f"问题:{case['question']}\n"
        f"参考答案:{case['reference']}\n"
        f"候选答案:{case['answer']}\n\n"
        f"评分维度:\n{crit_lines}\n\n"
        f'只输出 JSON,字段:reason(简短理由)、per_criterion(对象,键为 [{keys}],'
        f"值为 0~1 的数字)、overall(0~1 的总分)。"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


async def score_one(
    model: str, case: dict[str, Any], rubric: Rubric = DEFAULT_RUBRIC
) -> JurorScore:
    obj = await structured_chat(
        model,
        _build_messages(case, rubric),
        schema={"required": ["per_criterion", "overall"]},
        temperature=0.0,
        session_id="anvil-council",
    )
    raw = obj.get("per_criterion") or {}
    per_criterion = {c.key: float(raw.get(c.key, 0.0)) for c in rubric.criteria}
    return JurorScore(
        model=model,
        per_criterion=per_criterion,
        overall=float(obj.get("overall", 0.0)),
        reason=str(obj.get("reason", "")),
    )


async def score_case(
    case: dict[str, Any], models: list[str], rubric: Rubric = DEFAULT_RUBRIC
) -> list[JurorScore]:
    """Fan out: every juror scores the same case in parallel, independently."""
    return list(await asyncio.gather(*(score_one(m, case, rubric) for m in models)))
