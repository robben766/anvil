"""LLM-as-judge 基建:rubric + 先理由后分 + 结构化 JSON 输出 + 坏 JSON 重试一次。

judge 调用走 anvil-gateway —— 评测自身的成本被记账、链路被追踪(吃自己的狗粮)。
"""

from __future__ import annotations

import json
from typing import Any

from anvil_gateway import chat

JUDGE_MODEL = "deepseek-chat"

_SYSTEM = (
    "你是严格的评测裁判。先在 reason 字段里给出推理,再给结论字段。"
    "只输出一个 JSON 对象,不要输出任何其他文字。"
)


def _parse(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


async def judge_json(instruction: str, payload: dict[str, Any]) -> dict[str, Any]:
    """instruction = 评分规则(rubric);payload = 待评对象。返回解析后的 JSON。"""
    user = f"{instruction}\n\n待评对象:\n{json.dumps(payload, ensure_ascii=False)}"
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for _ in range(2):  # 1 次 + 坏 JSON 重试 1 次
        resp = await chat(JUDGE_MODEL, messages, temperature=0.0, session_id="anvil-eval")
        try:
            return _parse(resp.content or "")
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
    raise ValueError(f"judge output is not valid JSON after retry: {last_err}")
