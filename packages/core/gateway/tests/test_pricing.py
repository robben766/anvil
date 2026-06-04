from decimal import Decimal

from anvil_gateway.pricing import PRICES, compute_cost


def test_compute_cost_hand_calculated():
    """手算对照:prompt=1_000_000(其中 cached=500_000), completion=100_000。
    fresh 50万×input + cached 50万×cached + out 10万×output,再除以 1M。"""
    p = PRICES["deepseek-chat"]
    expected = (
        Decimal(500_000) * p["input"]
        + Decimal(500_000) * p["cached"]
        + Decimal(100_000) * p["output"]
    ) / Decimal(1_000_000)
    assert compute_cost("deepseek-chat", 1_000_000, 100_000, 500_000) == expected


def test_unknown_model_costs_zero():
    assert compute_cost("nope", 100, 100, 0) == Decimal(0)


def test_all_priced_models_have_three_fields():
    for model, p in PRICES.items():
        assert set(p) == {"input", "output", "cached"}, model


def test_served_model_name_resolves_price():
    """服务端返回实际模型名(deepseek-chat 映射为 deepseek-v4-flash),必须能计价。"""
    assert compute_cost("deepseek-v4-flash", 1_000_000, 0, 0) > Decimal(0)


def test_snapshot_name_prefix_match():
    """百炼快照名(如 qwen-plus-2026-01-25)按最长前缀匹配到 qwen-plus。"""
    assert compute_cost("qwen-plus-2026-01-25", 1_000_000, 0, 0) == compute_cost(
        "qwen-plus", 1_000_000, 0, 0
    )
