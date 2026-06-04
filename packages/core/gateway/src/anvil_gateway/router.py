"""路由:别名→候选模型链;冷却熔断;指数退避重试。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from anvil_gateway.errors import RetryableError

ALIASES: dict[str, list[str]] = {
    # 能力等价组:同组内可互为 fallback,不跨组降级
    "chat-default": ["deepseek-chat", "qwen-plus"],
}

MODEL_PROVIDER: dict[str, str] = {
    "deepseek-chat": "deepseek",
    "qwen-plus": "dashscope",
}

T = TypeVar("T")


def resolve(model: str) -> list[str]:
    """别名 → 候选模型列表;具体模型名 → 单元素列表。"""
    return list(ALIASES.get(model, [model]))


class Cooldown:
    """FatalAuth 后冷却 provider;到期自动恢复(半开)。"""

    def __init__(self, seconds: float = 300.0) -> None:
        self._seconds = seconds
        self._until: dict[str, float] = {}

    def mark(self, provider: str) -> None:
        self._until[provider] = time.monotonic() + self._seconds

    def available(self, provider: str) -> bool:
        return time.monotonic() >= self._until.get(provider, 0.0)


async def call_with_retry[T](
    fn: Callable[[], Awaitable[T]], *, max_retries: int = 2, base_delay: float = 0.5
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except RetryableError:
            if attempt >= max_retries:
                raise
            await asyncio.sleep(base_delay * (2**attempt))
            attempt += 1
