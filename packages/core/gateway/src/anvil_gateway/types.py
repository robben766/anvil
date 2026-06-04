"""公共类型。messages 直接用 OpenAI Chat Completions dict 格式,不另造抽象(YAGNI)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anvil_gateway.usage import UsageRecord

Message = dict[str, Any]


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str
    model: str
    provider: str
    usage: UsageRecord
    raw: dict[str, Any]


@dataclass
class ChatChunk:
    delta: str
    finish_reason: str | None = None
    usage: UsageRecord | None = None
