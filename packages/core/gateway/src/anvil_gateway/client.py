"""chat() 公共入口:路由 → 适配器 → 重试/fallback → 记账。"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from anvil_gateway.adapters.base import OpenAICompatAdapter
from anvil_gateway.adapters.dashscope import DashScopeAdapter
from anvil_gateway.adapters.deepseek import DeepSeekAdapter
from anvil_gateway.errors import (
    AllProvidersFailedError,
    FatalAuthError,
    FatalRequestError,
    RetryableError,
)
from anvil_gateway.ledger import SqliteLedger
from anvil_gateway.router import MODEL_PROVIDER, Cooldown, call_with_retry, resolve
from anvil_gateway.types import ChatResponse, Message

_ADAPTERS: dict[str, OpenAICompatAdapter] = {
    "deepseek": DeepSeekAdapter(),
    "dashscope": DashScopeAdapter(),
}
_cooldown = Cooldown()
_config: dict[str, Any] = {
    "ledger_path": os.environ.get("ANVIL_LEDGER_PATH", "anvil_ledger.sqlite3"),
    "timeout": 60.0,
    "retry_base_delay": 0.5,
}
_ledger: SqliteLedger | None = None


def configure(**kwargs: Any) -> None:
    """覆盖运行配置(ledger_path / timeout / retry_base_delay);重置 ledger 单例。"""
    global _ledger
    _config.update(kwargs)
    _ledger = None


def _get_ledger() -> SqliteLedger:
    global _ledger
    if _ledger is None:
        _ledger = SqliteLedger(_config["ledger_path"])
    return _ledger


async def _call_one(
    adapter: OpenAICompatAdapter,
    api_key: str,
    model: str,
    messages: list[Message],
    params: dict[str, Any],
    session_id: str | None,
) -> ChatResponse:
    payload = adapter.build_payload(model, messages, **params)
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=_config["timeout"]) as client:
        data = await adapter.send(client, api_key, payload)
    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = adapter.parse_usage(data, latency_ms=latency_ms, session_id=session_id)
    _get_ledger().insert(usage)
    return adapter.parse_response(data, usage)


async def chat(
    model: str,
    messages: list[Message],
    *,
    session_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> ChatResponse:
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
        "response_format": response_format,
    }
    failures: list[Exception] = []
    for candidate in resolve(model):
        provider = MODEL_PROVIDER.get(candidate)
        if provider is None:
            raise FatalRequestError(f"unknown model: {candidate}")
        if not _cooldown.available(provider):
            continue
        adapter = _ADAPTERS[provider]
        api_key = os.environ.get(adapter.api_key_env, "")
        try:
            return await call_with_retry(
                lambda a=adapter, k=api_key, m=candidate: _call_one(
                    a, k, m, messages, params, session_id
                ),
                base_delay=_config["retry_base_delay"],
            )
        except RetryableError as e:
            failures.append(e)
        except FatalAuthError as e:
            _cooldown.mark(provider)
            failures.append(e)
    raise AllProvidersFailedError(f"all candidates failed: {failures!r}")
