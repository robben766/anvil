import json

import httpx
import pytest
import respx
from anvil_eval.judge import judge_json

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _ok(payload: dict):
    return httpx.Response(
        200,
        json={
            "id": "j1",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        },
    )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    from anvil_gateway import configure

    configure(
        database_url=__import__("os").environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


@respx.mock
async def test_judge_returns_parsed_json(env):
    respx.post(DS_URL).mock(return_value=_ok({"reason": "因为…", "verdict": True}))
    out = await judge_json("判断主张是否被支持", {"claim": "x", "context": "y"})
    assert out == {"reason": "因为…", "verdict": True}


@respx.mock
async def test_judge_strips_markdown_fence(env):
    fenced = '```json\n{"reason": "r", "verdict": false}\n```'
    resp = httpx.Response(
        200,
        json={
            "id": "j1",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": fenced},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        },
    )
    respx.post(DS_URL).mock(return_value=resp)
    out = await judge_json("instr", {})
    assert out["verdict"] is False


@respx.mock
async def test_judge_retries_once_on_bad_json(env):
    bad = httpx.Response(
        200,
        json={
            "id": "j",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "not json"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    route = respx.post(DS_URL)
    route.side_effect = [bad, _ok({"reason": "r", "verdict": True})]
    out = await judge_json("instr", {})
    assert out["verdict"] is True
    assert route.call_count == 2


@respx.mock
async def test_judge_gives_up_after_retry(env):
    bad = httpx.Response(
        200,
        json={
            "id": "j",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "still not json"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    respx.post(DS_URL).mock(return_value=bad)
    with pytest.raises(ValueError, match="judge"):
        await judge_json("instr", {})
