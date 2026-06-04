import httpx
import pytest
import respx

import anvil_gateway.client as client_mod
from anvil_gateway import chat, configure
from anvil_gateway.errors import AllProvidersFailedError, FatalRequestError
from anvil_gateway.ledger import SqliteLedger
from anvil_gateway.router import Cooldown

DS_URL = "https://api.deepseek.com/v1/chat/completions"
QW_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

DS_OK = {
    "id": "chatcmpl-ds",
    "model": "deepseek-chat",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "你好"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_cache_hit_tokens": 4},
}

QW_OK = {
    "id": "chatcmpl-qw",
    "model": "qwen-plus",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "好的"}, "finish_reason": "stop"}
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 6,
        "prompt_tokens_details": {"cached_tokens": 0},
    },
}

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def env_and_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k2")
    path = str(tmp_path / "ledger.sqlite3")
    configure(ledger_path=path, retry_base_delay=0)
    client_mod._cooldown = Cooldown()
    yield path


@respx.mock
async def test_happy_path_records_usage(env_and_ledger):
    respx.post(DS_URL).mock(return_value=httpx.Response(200, json=DS_OK))
    resp = await chat("deepseek-chat", MSGS, session_id="s-1")
    assert resp.content == "你好" and resp.provider == "deepseek"
    assert resp.usage.cached_tokens == 4 and resp.usage.cache_hit_rate == 0.4
    ledger = SqliteLedger(env_and_ledger)
    assert ledger.count() == 1


@respx.mock
async def test_fallback_to_second_provider(env_and_ledger):
    ds_route = respx.post(DS_URL).mock(return_value=httpx.Response(500, text="down"))
    respx.post(QW_URL).mock(return_value=httpx.Response(200, json=QW_OK))
    resp = await chat("chat-default", MSGS)
    assert resp.provider == "dashscope" and resp.content == "好的"
    assert ds_route.call_count == 3  # 1 次 + 2 重试


@respx.mock
async def test_fatal_request_raises_immediately(env_and_ledger):
    route = respx.post(DS_URL).mock(return_value=httpx.Response(400, text="bad param"))
    with pytest.raises(FatalRequestError):
        await chat("deepseek-chat", MSGS)
    assert route.call_count == 1  # 不重试


@respx.mock
async def test_all_failed(env_and_ledger):
    respx.post(DS_URL).mock(return_value=httpx.Response(500))
    respx.post(QW_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(AllProvidersFailedError):
        await chat("chat-default", MSGS)


async def test_unknown_model_is_fatal(env_and_ledger):
    with pytest.raises(FatalRequestError):
        await chat("gpt-99", MSGS)


@respx.mock
async def test_auth_failure_triggers_cooldown_then_fallback(env_and_ledger):
    ds_route = respx.post(DS_URL).mock(return_value=httpx.Response(401, text="unauthorized"))
    respx.post(QW_URL).mock(return_value=httpx.Response(200, json=QW_OK))
    resp = await chat("chat-default", MSGS)
    assert resp.provider == "dashscope"
    assert ds_route.call_count == 1  # FatalAuth 不重试,直接冷却切换
    # 第二次调用:deepseek 仍在冷却,直接走 dashscope,deepseek 不再被请求
    resp2 = await chat("chat-default", MSGS)
    assert resp2.provider == "dashscope"
    assert ds_route.call_count == 1
