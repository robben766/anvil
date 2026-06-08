import json
import os

import httpx
import pytest
import respx
from anvil_guard.structured import StructuredOutputError, structured_chat

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _resp(content: str):
    return httpx.Response(
        200,
        json={
            "id": "s1",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


@respx.mock
async def test_returns_parsed_object():
    respx.post(DS_URL).mock(return_value=_resp('{"a": 1, "b": "x"}'))
    out = await structured_chat("deepseek-chat", [{"role": "user", "content": "json please"}])
    assert out == {"a": 1, "b": "x"}


@respx.mock
async def test_strips_markdown_fence():
    respx.post(DS_URL).mock(return_value=_resp('```json\n{"ok": true}\n```'))
    out = await structured_chat("deepseek-chat", [{"role": "user", "content": "json"}])
    assert out["ok"] is True


@respx.mock
async def test_validates_required_keys_and_retries():
    route = respx.post(DS_URL)
    route.side_effect = [_resp('{"reason": "r"}'), _resp('{"reason": "r", "verdict": true}')]
    out = await structured_chat(
        "deepseek-chat",
        [{"role": "user", "content": "json"}],
        schema={"required": ["reason", "verdict"]},
    )
    assert out == {"reason": "r", "verdict": True}
    assert route.call_count == 2


@respx.mock
async def test_raises_after_retries_exhausted():
    respx.post(DS_URL).mock(return_value=_resp("not json at all"))
    with pytest.raises(StructuredOutputError):
        await structured_chat("deepseek-chat", [{"role": "user", "content": "json"}])


@respx.mock
async def test_rejects_non_object_json():
    respx.post(DS_URL).mock(return_value=_resp("[1, 2, 3]"))
    with pytest.raises(StructuredOutputError):
        await structured_chat("deepseek-chat", [{"role": "user", "content": "json"}], max_retries=0)


@respx.mock
async def test_none_content_raises():
    # Model returns a message with no content → must raise, not crash on None.
    resp = httpx.Response(
        200,
        json={
            "id": "s1",
            "model": "deepseek-chat",
            "choices": [
                {"index": 0, "message": {"role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        },
    )
    respx.post(DS_URL).mock(return_value=resp)
    with pytest.raises(StructuredOutputError):
        await structured_chat("deepseek-chat", [{"role": "user", "content": "json"}], max_retries=0)


@respx.mock
async def test_response_format_override_is_forwarded():
    route = respx.post(DS_URL).mock(return_value=_resp('{"ok": true}'))
    await structured_chat(
        "deepseek-chat",
        [{"role": "user", "content": "json"}],
        response_format={"type": "text"},
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"] == {"type": "text"}
