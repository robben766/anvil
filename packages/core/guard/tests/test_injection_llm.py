import json
import os

import httpx
import pytest
import respx
from anvil_guard.injection import detect_injection_llm

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _resp(obj: dict):
    return httpx.Response(
        200,
        json={
            "id": "i1",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(obj, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
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
async def test_llm_flags_injection():
    respx.post(DS_URL).mock(
        return_value=_resp({"is_injection": True, "category": "jailbreak", "reason": "角色越权"})
    )
    v = await detect_injection_llm("一段语义上越权但无明显关键词的文本")
    assert v.is_injection is True
    assert v.category == "jailbreak"


@respx.mock
async def test_llm_passes_benign():
    respx.post(DS_URL).mock(
        return_value=_resp({"is_injection": False, "category": "none", "reason": "正常提问"})
    )
    v = await detect_injection_llm("等待期是多少天?")
    assert v.is_injection is False
    assert v.category == "none"
