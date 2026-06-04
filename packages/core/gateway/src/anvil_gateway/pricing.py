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
    # 来源:https://help.aliyun.com/zh/model-studio/qwen-model-billing-notice (核对日期:2026-06-04)
    # 缓存命中(隐式缓存)= 输入标准价的 20%,来源:
    #   https://help.aliyun.com/zh/model-studio/context-cache
    # qwen-plus 输出官方刊例为 ¥2/M(¥0.002/千 token),计划占位值 8.0 有误。
    # 缓存命中计划占位值 0.32 有误,官方隐式缓存为输入价 ×20% = ¥0.16/M。
    "qwen-plus": {
        "input": Decimal("0.8"),
        "output": Decimal("2"),
        "cached": Decimal("0.16"),
    },
}

_MILLION = Decimal(1_000_000)


def compute_cost(
    model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int
) -> Decimal:
    prices = PRICES.get(model)
    if prices is None:
        return Decimal(0)
    fresh = Decimal(prompt_tokens - cached_tokens)
    return (
        fresh * prices["input"]
        + Decimal(cached_tokens) * prices["cached"]
        + Decimal(completion_tokens) * prices["output"]
    ) / _MILLION
