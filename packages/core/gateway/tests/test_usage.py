from decimal import Decimal

from anvil_gateway.usage import UsageRecord


def _record(prompt=100, cached=30):
    return UsageRecord(
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=prompt,
        completion_tokens=50,
        cached_tokens=cached,
        cost_cny=Decimal("0.001"),
        latency_ms=800,
        request_id="req-1",
    )


def test_cache_hit_rate():
    assert _record(prompt=100, cached=30).cache_hit_rate == 0.3


def test_cache_hit_rate_zero_prompt():
    assert _record(prompt=0, cached=0).cache_hit_rate == 0.0


def test_defaults():
    r = _record()
    assert r.ttft_ms is None and r.session_id is None
    assert r.created_at.tzinfo is not None  # 必须带时区
