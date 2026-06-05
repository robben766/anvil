import httpx
import pytest
import respx

import anvil_gateway.client as client_mod
from anvil_gateway import configure
from anvil_gateway.proxy.app import app
from anvil_gateway.router import Cooldown

DS_URL = "https://api.deepseek.com/v1/chat/completions"

DS_OK = {
    "id": "chatcmpl-ds",
    "model": "deepseek-chat",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "你好"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_cache_hit_tokens": 4},
}

BODY = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture(autouse=True)
def env(monkeypatch, pg_ledger):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.delenv("ANVIL_PROXY_API_KEY", raising=False)
    import os

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )
    client_mod._cooldown = Cooldown()


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@respx.mock
async def test_openai_compatible_response(env):
    respx.post(DS_URL).mock(return_value=httpx.Response(200, json=DS_OK))
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "你好"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    assert data["model"] == "deepseek-chat"


@respx.mock
async def test_upstream_fatal_maps_to_400(env):
    respx.post(DS_URL).mock(return_value=httpx.Response(400, text="bad param"))
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 400
    assert "error" in r.json()


@respx.mock
async def test_all_providers_failed_maps_to_502(env):
    respx.post(DS_URL).mock(return_value=httpx.Response(500))
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 502


async def test_auth_required_when_key_set(env, monkeypatch):
    monkeypatch.setenv("ANVIL_PROXY_API_KEY", "secret-1")
    async with _client() as c:
        r = await c.post("/v1/chat/completions", json=BODY)
        assert r.status_code == 401
        r2 = await c.post(
            "/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer wrong"}
        )
        assert r2.status_code == 401
