"""价表与成本折算。成本为估算值;价格变更须更新核对日期并走 ADR。"""

from __future__ import annotations

from decimal import Decimal

# 单位:元 / 百万 token。核对日期:2026-06-04(来源见各条注释)
PRICES: dict[str, dict[str, Decimal]] = {
    # 来源:https://api-docs.deepseek.com/zh-cn/quick_start/pricing (核对日期:2026-06-04)
    # deepseek-chat 当前映射到 deepseek-v4-flash 非思考模式;模型名将于 2026-07-24 弃用。
    # 计划占位值(2.0/0.5/8.0)与官方刊例不符,已按官方页更正。
    "deepseek-chat": {
        "input": Decimal("1"),
        "output": Decimal("2"),
        "cached": Decimal("0.02"),
    },
    # 服务端对 deepseek-chat 实际返回的模型名(live 验证发现),价格同上
    "deepseek-v4-flash": {
        "input": Decimal("1"),
        "output": Decimal("2"),
        "cached": Decimal("0.02"),
    },
    # 来源:https://help.aliyun.com/zh/model-studio/qwen-model-billing-notice (核对日期:2026-06-04)
    # 缓存命中(隐式缓存)= 输入标准价的 20%,来源:
    #   https://help.aliyun.com/zh/model-studio/context-cache
    # qwen-plus 实时调用输出 ¥0.0004/千 token = ¥0.4/M(2024-09-19 生效刊例,
    # 2026-06-04 复核;勿与 batch 价混淆)。缓存命中 = 隐式缓存,输入价 ×20% = ¥0.16/M。
    "qwen-plus": {
        "input": Decimal("0.8"),
        "output": Decimal("0.4"),
        "cached": Decimal("0.16"),
    },
}

_MILLION = Decimal(1_000_000)


def _lookup(model: str) -> dict[str, Decimal] | None:
    """先精确匹配;否则按最长前缀匹配(服务端常返回快照名,如 qwen-plus-2026-01-25)。"""
    prices = PRICES.get(model)
    if prices is not None:
        return prices
    candidates = [k for k in PRICES if model.startswith(k)]
    if candidates:
        return PRICES[max(candidates, key=len)]
    return None


def compute_cost(
    model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int
) -> Decimal:
    prices = _lookup(model)
    if prices is None:
        return Decimal(0)
    fresh = Decimal(prompt_tokens - cached_tokens)
    return (
        fresh * prices["input"]
        + Decimal(cached_tokens) * prices["cached"]
        + Decimal(completion_tokens) * prices["output"]
    ) / _MILLION
