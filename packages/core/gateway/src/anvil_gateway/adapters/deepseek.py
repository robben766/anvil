"""DeepSeek:缓存命中在 usage 顶层 prompt_cache_hit_tokens。"""

from __future__ import annotations

from typing import Any

from anvil_gateway.adapters.base import OpenAICompatAdapter


class DeepSeekAdapter(OpenAICompatAdapter):
    provider = "deepseek"
    base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"

    def parse_cached_tokens(self, usage: dict[str, Any]) -> int:
        return usage.get("prompt_cache_hit_tokens", 0)
