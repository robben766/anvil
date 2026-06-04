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
