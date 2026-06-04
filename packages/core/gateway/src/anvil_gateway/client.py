"""chat() 公共入口:路由 → 适配器 → 重试/fallback → 记账。"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
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
    classify_status,
)
from anvil_gateway.ledger import PostgresLedger
from anvil_gateway.router import MODEL_PROVIDER, Cooldown, call_with_retry, resolve
from anvil_gateway.types import ChatChunk, ChatResponse, Message

# NOTE: 模块级单例(进程内)——多产品共享同一配置/冷却/账本;需要隔离时再引入 Gateway 类。
_ADAPTERS: dict[str, OpenAICompatAdapter] = {
    "deepseek": DeepSeekAdapter(),
    "dashscope": DashScopeAdapter(),
}
_cooldown = Cooldown()
_config: dict[str, Any] = {
    "database_url": os.environ.get("ANVIL_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5433/anvil"),
    "timeout": 60.0,
    "retry_base_delay": 0.5,
}
_ledger: PostgresLedger | None = None


def configure(**kwargs: Any) -> None:
    """覆盖运行配置(database_url / timeout / retry_base_delay);重置 ledger 单例。"""
    global _ledger
    _config.update(kwargs)
    _ledger = None


def _get_ledger() -> PostgresLedger:
    global _ledger
    if _ledger is None:
        _ledger = PostgresLedger(_config["database_url"])
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
    # TODO: 复用 per-provider AsyncClient 连接池;低 QPS 下可接受。
    async with httpx.AsyncClient(timeout=_config["timeout"]) as client:
        data = await adapter.send(client, api_key, payload)
    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = adapter.parse_usage(data, latency_ms=latency_ms, session_id=session_id)
    # NOTE: PG 写入为 async,不再阻塞 event loop
    await _get_ledger().insert(usage)
    return adapter.parse_response(data, usage)


async def _stream_one(
    adapter: OpenAICompatAdapter,
    api_key: str,
    model: str,
    messages: list[Message],
    params: dict[str, Any],
    session_id: str | None,
) -> AsyncIterator[ChatChunk]:
    payload = adapter.build_payload(model, messages, **params)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    ttft_ms: int | None = None
    try:
        async with httpx.AsyncClient(timeout=_config["timeout"]) as client:
            async with client.stream(
                "POST",
                f"{adapter.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread())[:200]
                    raise classify_status(resp.status_code)(
                        f"{adapter.provider} HTTP {resp.status_code}: {body!r}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):]
                    if raw == "[DONE]":
                        break
                    chunk = json.loads(raw)
                    if chunk.get("usage"):
                        latency_ms = int((time.perf_counter() - start) * 1000)
                        usage = adapter.parse_usage(
                            chunk, latency_ms=latency_ms, ttft_ms=ttft_ms, session_id=session_id
                        )
                        await _get_ledger().insert(usage)
                        # usage 末块按约定为流的最后一个数据块,finish_reason 统一记 "stop"
                        yield ChatChunk(delta="", finish_reason="stop", usage=usage)
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta and ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - start) * 1000)
                    finish = choices[0].get("finish_reason")
                    if delta or finish:
                        yield ChatChunk(delta=delta, finish_reason=finish)
    except httpx.TimeoutException as e:
        raise RetryableError(f"{adapter.provider} stream timeout: {e}") from e
    except httpx.TransportError as e:
        # NOTE: 流中途断开无法续传,统一归类为 RetryableError,由调用方决定是否整体重发
        raise RetryableError(f"{adapter.provider} stream transport: {e}") from e


async def chat(
    model: str,
    messages: list[Message],
    *,
    stream: bool = False,
    session_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> ChatResponse | AsyncIterator[ChatChunk]:
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
        "response_format": response_format,
    }
    if stream:
        candidate = resolve(model)[0]  # 薄版:流式不做跨 provider fallback
        provider = MODEL_PROVIDER.get(candidate)
        if provider is None:
            raise FatalRequestError(f"unknown model: {candidate}")
        adapter = _ADAPTERS[provider]
        api_key = os.environ.get(adapter.api_key_env, "")
        return _stream_one(adapter, api_key, candidate, messages, params, session_id)
    failures: list[Exception] = []
    for candidate in resolve(model):
        provider = MODEL_PROVIDER.get(candidate)
        if provider is None:
            raise FatalRequestError(f"unknown model: {candidate}")
        if not _cooldown.available(provider):
            failures.append(RetryableError(f"{provider} on cooldown"))
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
