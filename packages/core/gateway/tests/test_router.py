import pytest

from anvil_gateway.errors import RetryableError
from anvil_gateway.router import ALIASES, MODEL_PROVIDER, Cooldown, call_with_retry, resolve


def test_resolve_alias_and_specific():
    assert resolve("chat-default") == ["deepseek-chat", "qwen-plus"]
    assert resolve("deepseek-chat") == ["deepseek-chat"]


def test_alias_models_all_have_provider():
    for models in ALIASES.values():
        for m in models:
            assert m in MODEL_PROVIDER


def test_cooldown_marks_and_recovers():
    cd = Cooldown(seconds=0.0)  # 立即过期 = 标记后下一刻即恢复(半开)
    assert cd.available("deepseek")
    cd.mark("deepseek")
    assert cd.available("deepseek")
    cd2 = Cooldown(seconds=60.0)
    cd2.mark("deepseek")
    assert not cd2.available("deepseek")
    assert cd2.available("dashscope")


async def test_retry_succeeds_after_failures():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("again")
        return "ok"

    assert await call_with_retry(flaky, max_retries=2, base_delay=0) == "ok"
    assert calls["n"] == 3


async def test_retry_exhausted_raises():
    async def always_fail():
        raise RetryableError("nope")

    with pytest.raises(RetryableError):
        await call_with_retry(always_fail, max_retries=2, base_delay=0)
