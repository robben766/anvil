"""LLM-as-judge 基建:rubric + 先理由后分 + 结构化 JSON 输出。

解析/重试逻辑已收敛到 anvil_guard.structured_chat(安全竖梁的结构化输出能力),
judge 只负责拼 rubric prompt 并复用它(吃自己的狗粮:评测依赖通用底座)。
"""

from __future__ import annotations

import json
from typing import Any

from anvil_guard import StructuredOutputError, structured_chat

JUDGE_MODEL = "deepseek-chat"

_SYSTEM = (
    "你是严格的评测裁判。先在 reason 字段里给出推理,再给结论字段。"
    "只输出一个 JSON 对象,不要输出任何其他文字。"
)


async def judge_json(instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
    """instruction = 评分规则(rubric);payload = 待评对象。返回解析后的 JSON。"""
    user = f"{instruction}\n\n待评对象:\n{json.dumps(payload, ensure_ascii=False)}"
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
    try:
        return await structured_chat(
            JUDGE_MODEL, messages, temperature=0.0, session_id="anvil-eval"
        )
    except StructuredOutputError as e:
        raise ValueError(f"judge output is not valid JSON after retry: {e}") from e
