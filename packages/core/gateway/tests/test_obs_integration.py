"""Integration tests: every chat() call emits a GenAI OTEL span."""
import os

import httpx
import pytest
import respx
from anvil_obs import semconv
from anvil_obs.exporter import OtlpExporter, drain_for_test

import anvil_gateway.client as client_mod
from anvil_gateway import chat, configure
from anvil_gateway.errors import FatalRequestError
from anvil_gateway.router import Cooldown

TEST_DB_URL = os.environ.get(
    "ANVIL_TEST_DATABASE_URL", "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test"
)

DS_URL = "https://api.deepseek.com/v1/chat/completions"
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

_D = "deepseek-chat"

DS_OK = {
    "id": "c1",
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "好"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_cache_hit_tokens": 4,
    },
}

DASHSCOPE_OK = {
    "id": "d1",
    "model": "qwen-plus",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "好"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_cache_hit_tokens": 0,
    },
}

# SSE payload for streaming test (matches test_client_stream.py pattern)
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
    '"prompt_cache_hit_tokens":4}}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture(autouse=True)
async def env(monkeypatch, pg_ledger):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k2")
    configure(database_url=TEST_DB_URL, retry_base_delay=0)
    client_mod._cooldown = Cooldown()
    exp = OtlpExporter(
        endpoint="http://obs.invalid/v1/traces",
        public_key="pk",
        secret_key="sk",
        flush_interval=9999,
    )
    yield
    await exp.aclose()


@respx.mock
async def test_chat_emits_genai_span():
    respx.post(DS_URL).mock(return_value=httpx.Response(200, json=DS_OK))
    await chat("deepseek-chat", [{"role": "user", "content": "hi"}], session_id="s1")
    spans = drain_for_test()
    s = next(x for x in spans if x.name.startswith("chat "))
    a = s.attributes
    assert a[semconv.GEN_AI_SYSTEM] == "deepseek"
    assert a[semconv.GEN_AI_REQUEST_MODEL] == "deepseek-chat"
    assert a[semconv.GEN_AI_RESPONSE_MODEL] == "deepseek-chat"
    assert a[semconv.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert a[semconv.GEN_AI_USAGE_OUTPUT_TOKENS] == 2
    assert a[semconv.ANVIL_CACHED_TOKENS] == 4
    assert a[semconv.ANVIL_CACHE_HIT_RATE] == pytest.approx(0.4)
    assert isinstance(a[semconv.ANVIL_COST_CNY], float)
    assert a[semconv.ANVIL_SESSION_ID] == "s1"
    assert s.status_ok


@respx.mock
async def test_failed_chat_marks_span_error():
    respx.post(DS_URL).mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(FatalRequestError):
        await chat("deepseek-chat", [{"role": "user", "content": "hi"}])
    spans = drain_for_test()
    s = next(x for x in spans if x.name.startswith("chat "))
    assert not s.status_ok
    assert "400" in s.status_message


@respx.mock
async def test_stream_emits_span_with_ttft():
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
    assert chunks  # stream produced data
    spans = drain_for_test()
    s = next(x for x in spans if x.name.startswith("chat "))
    assert s is not None
    assert semconv.ANVIL_TTFT_MS in s.attributes
    assert s.status_ok


@respx.mock
async def test_root_span_wraps_fallback_attempts():
    """gateway.chat root span wraps retry/fallback child spans into one tree."""
    # DeepSeek fails 3 times (mock: always 500), DashScope succeeds
    respx.post(DS_URL).mock(return_value=httpx.Response(500, text="err"))
    respx.post(DASHSCOPE_URL).mock(return_value=httpx.Response(200, json=DASHSCOPE_OK))

    await chat("chat-default", [{"role": "user", "content": "hi"}])
    spans = drain_for_test()

    # There must be a root span named "gateway.chat chat-default"
    roots = [s for s in spans if s.name == "gateway.chat chat-default"]
    assert roots, f"no gateway.chat root span found; spans={[s.name for s in spans]}"
    root = roots[0]

    # All "chat " child spans must have root as their parent
    chat_children = [s for s in spans if s.name.startswith("chat ")]
    assert chat_children, "expected at least one chat child span"
    for child in chat_children:
        assert child.parent_span_id == root.span_id, (
            f"child '{child.name}' parent {child.parent_span_id!r} != root {root.span_id!r}"
        )
