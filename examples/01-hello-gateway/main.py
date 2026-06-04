"""最小示例:一个 chat() 调用任意 provider,自动 fallback 与记账。

连续运行两次,观察第二次 cached 命中:
- DeepSeek 磁盘缓存按 64-token 块做前缀匹配,prompt 太短永远不命中——
  所以这里放一段固定的长 system prompt(静态前缀放最前,这是缓存工程的铁律)。
"""

import asyncio

from anvil_gateway import chat

# 固定不变的 system prompt(约 200 token):它就是可被缓存复用的"静态前缀"
SYSTEM = (
    "你是 anvil 项目的技术讲解助手。anvil 是一个 AI 工程深度学习项目,"
    "自研实现 LLM 应用的核心模块:统一模型网关(多 provider 适配、路由与故障切换、"
    "成本与缓存命中记账)、OTEL 标准可观测、RAGAS 评测体系。"
    "回答要求:面向有经验的后端工程师,用准确的技术语言,"
    "先给一句话结论,再给一段不超过三句话的展开说明,不使用比喻,"
    "不重复问题本身,不输出列表,保持克制与精确。"
)


async def main() -> None:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "用一句话解释什么是模型网关"},
    ]
    resp = await chat("chat-default", messages)
    print(f"[{resp.provider}/{resp.model}] {resp.content}")
    u = resp.usage
    print(
        f"tokens={u.prompt_tokens}+{u.completion_tokens}"
        f" cached={u.cached_tokens}({u.cache_hit_rate:.0%}) cost≈¥{u.cost_cny}"
    )


asyncio.run(main())
