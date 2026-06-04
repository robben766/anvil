import httpx
import pytest
import respx

from anvil_gateway.adapters.base import OpenAICompatAdapter
from anvil_gateway.errors import FatalAuthError, FatalRequestError, RetryableError


class FakeAdapter(OpenAICompatAdapter):
    provider = "fake"
    base_url = "https://fake.example.com/v1"
    api_key_env = "FAKE_KEY"

    def parse_cached_tokens(self, usage):
        return usage.get("cached", 0)


OK_DATA = {
    "id": "chatcmpl-1",
    "model": "fake-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cached": 4},
}


def test_build_payload_drops_none():
    p = FakeAdapter().build_payload(
        "m", [{"role": "user", "content": "x"}], temperature=None, max_tokens=8
    )
    assert p == {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8}


def test_parse_usage_and_response():
    a = FakeAdapter()
    usage = a.parse_usage(OK_DATA, latency_ms=123)
    assert usage.provider == "fake"
    assert usage.prompt_tokens == 10 and usage.cached_tokens == 4
    assert usage.request_id == "chatcmpl-1"
    resp = a.parse_response(OK_DATA, usage)
    assert resp.content == "hi" and resp.finish_reason == "stop" and resp.usage is usage


@pytest.mark.parametrize(
    ("status", "exc"),
    [(429, RetryableError), (500, RetryableError), (401, FatalAuthError), (400, FatalRequestError)],
)
@respx.mock
async def test_send_classifies_http_errors(status, exc):
    respx.post("https://fake.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(status, text="boom")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(exc):
            await FakeAdapter().send(client, "k", {"model": "m", "messages": []})


@respx.mock
async def test_send_timeout_is_retryable():
    respx.post("https://fake.example.com/v1/chat/completions").mock(
        side_effect=httpx.ConnectTimeout("t")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(RetryableError):
            await FakeAdapter().send(client, "k", {"model": "m", "messages": []})
