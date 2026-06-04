import os

import httpx
import pytest
import respx

import anvil_gateway.client as client_mod
from anvil_gateway import chat, configure
from anvil_gateway.router import Cooldown

TEST_DB_URL = os.environ.get(
    "ANVIL_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5433/anvil"
)

DS_URL = "https://api.deepseek.com/v1/chat/completions"

_D = "deepseek-chat"
SSE = (
    'data: {"id":"c1","model":"' + _D + '",'
    '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","model":"' + _D + '",'
    '"choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","model":"' + _D + '",'
    '"choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","model":"' + _D + '",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"id":"c1","model":"' + _D + '","choices":[],'
    '"usage":{"prompt_tokens":10,"completion_tokens":2,'
    '"prompt_cache_hit_tokens":8}}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture(autouse=True)
async def env_and_ledger(monkeypatch, pg_ledger):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    configure(database_url=TEST_DB_URL, retry_base_delay=0)
    client_mod._cooldown = Cooldown()
    yield pg_ledger


@respx.mock
async def test_stream_yields_deltas_and_final_usage(env_and_ledger):
    respx.post(DS_URL).mock(
        return_value=httpx.Response(
            200, text=SSE, headers={"content-type": "text/event-stream"}
        )
    )
    chunks = [
        c
        async for c in await chat(
            "deepseek-chat", [{"role": "user", "content": "hi"}], stream=True
        )
    ]
    text = "".join(c.delta for c in chunks)
    assert text == "你好"
    finals = [c for c in chunks if c.usage is not None]
    assert len(finals) == 1
    assert finals[0].usage.cached_tokens == 8
    assert finals[0].usage.ttft_ms is not None
    assert await env_and_ledger.count() == 1


@respx.mock
async def test_stream_transport_error_is_classified(env_and_ledger):
    respx.post(DS_URL).mock(side_effect=httpx.ReadError("connection lost"))
    from anvil_gateway.errors import RetryableError

    with pytest.raises(RetryableError):
        async for _ in await chat(
            "deepseek-chat", [{"role": "user", "content": "hi"}], stream=True
        ):
            pass
