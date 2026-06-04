"""OpenAI 兼容 provider 共用适配器;子类只覆盖 usage 缓存字段差异。"""

from __future__ import annotations

from typing import Any

import httpx

from anvil_gateway.errors import FatalRequestError, RetryableError, classify_status
from anvil_gateway.pricing import compute_cost
from anvil_gateway.types import ChatResponse
from anvil_gateway.usage import UsageRecord


class OpenAICompatAdapter:
    provider: str
    base_url: str
    api_key_env: str

    def parse_cached_tokens(self, usage: dict[str, Any]) -> int:
        raise NotImplementedError

    def build_payload(
        self, model: str, messages: list[dict[str, Any]], **params: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update({k: v for k, v in params.items() if v is not None})
        return payload

    async def send(
        self, client: httpx.AsyncClient, api_key: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except httpx.TimeoutException as e:
            raise RetryableError(f"{self.provider} timeout: {e}") from e
        except httpx.TransportError as e:
            raise RetryableError(f"{self.provider} transport: {e}") from e
        if resp.status_code != 200:
            raise classify_status(resp.status_code)(
                f"{self.provider} HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json()
        except ValueError as e:
            raise RetryableError(f"{self.provider} non-JSON 200: {resp.text[:200]}") from e

    def parse_usage(
        self,
        data: dict[str, Any],
        *,
        latency_ms: int,
        ttft_ms: int | None = None,
        session_id: str | None = None,
    ) -> UsageRecord:
        u = data.get("usage") or {}
        prompt = u.get("prompt_tokens", 0)
        completion = u.get("completion_tokens", 0)
        cached = self.parse_cached_tokens(u)
        model = data.get("model", "")
        return UsageRecord(
            provider=self.provider,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            cost_cny=compute_cost(model, prompt, completion, cached),
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            request_id=data.get("id", ""),
            session_id=session_id,
        )

    def parse_response(self, data: dict[str, Any], usage: UsageRecord) -> ChatResponse:
        choices = data.get("choices") or []
        if not choices:
            raise FatalRequestError(f"{self.provider} returned no choices")
        choice = choices[0]
        msg = choice["message"]
        return ChatResponse(
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            finish_reason=choice.get("finish_reason", ""),
            model=data.get("model", ""),
            provider=self.provider,
            usage=usage,
            raw=data,
        )
