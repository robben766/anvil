"""最小示例:一个 chat() 调用任意 provider,自动 fallback 与记账。"""

import asyncio

from anvil_gateway import chat


async def main() -> None:
    resp = await chat("chat-default", [{"role": "user", "content": "用一句话解释什么是模型网关"}])
    print(f"[{resp.provider}/{resp.model}] {resp.content}")
    u = resp.usage
    print(
        f"tokens={u.prompt_tokens}+{u.completion_tokens}"
        f" cached={u.cached_tokens}({u.cache_hit_rate:.0%}) cost≈¥{u.cost_cny}"
    )


asyncio.run(main())
