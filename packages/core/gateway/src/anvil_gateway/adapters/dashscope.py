"""DashScope(百炼)OpenAI 兼容模式:缓存命中在 prompt_tokens_details.cached_tokens。"""

from __future__ import annotations

from typing import Any

from anvil_gateway.adapters.base import OpenAICompatAdapter


class DashScopeAdapter(OpenAICompatAdapter):
    provider = "dashscope"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env = "DASHSCOPE_API_KEY"

    def parse_cached_tokens(self, usage: dict[str, Any]) -> int:
        return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
