import json
import os

import httpx
import pytest
import respx
from anvil_council.cli import _run_judge

DS_URL = "https://api.deepseek.com/v1/chat/completions"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def _resp(score: float, model: str):
    obj = {
        "reason": "r",
        "per_criterion": {
            k: score for k in ("correctness", "evidence", "completeness", "relevance")
        },
        "overall": score,
    }
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(obj)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k2")
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


@respx.mock
async def test_run_judge_prints_verdict(tmp_path, capsys):
    respx.post(DS_URL).mock(return_value=_resp(1.0, "deepseek-chat"))
    respx.post(QWEN_URL).mock(return_value=_resp(0.0, "qwen-plus"))
    ds = tmp_path / "cases.jsonl"
    ds.write_text(
        json.dumps({"question": "q", "reference": "r", "answer": "a"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    await _run_judge(str(ds), ["deepseek-chat", "qwen-plus"])
    out = capsys.readouterr().out
    assert "correctness" in out
    assert "分歧" in out or "disagree" in out.lower()
