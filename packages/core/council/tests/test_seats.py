import json
import os

import httpx
import pytest
import respx
from anvil_council.rubric import DEFAULT_RUBRIC
from anvil_council.seats import score_case, score_one
from anvil_council.verdict import JurorScore

DS_URL = "https://api.deepseek.com/v1/chat/completions"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

CASE = {
    "question": "等待期是多少天?",
    "reference": "本合同等待期为90天。",
    "answer": "等待期为90天。",
}


def _resp(obj: dict, model: str):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": model,
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
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _score_obj(score: float):
    return {
        "reason": "符合参考",
        "per_criterion": {
            "correctness": score,
            "evidence": score,
            "completeness": score,
            "relevance": score,
        },
        "overall": score,
    }


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
async def test_score_one_returns_juror_score():
    respx.post(DS_URL).mock(return_value=_resp(_score_obj(1.0), "deepseek-chat"))
    js = await score_one("deepseek-chat", CASE, DEFAULT_RUBRIC)
    assert isinstance(js, JurorScore)
    assert js.model == "deepseek-chat"
    assert js.per_criterion["correctness"] == 1.0
    assert js.overall == 1.0


@respx.mock
async def test_score_one_defaults_missing_criterion_to_zero():
    obj = {"reason": "r", "per_criterion": {"correctness": 1.0}, "overall": 0.5}
    respx.post(DS_URL).mock(return_value=_resp(obj, "deepseek-chat"))
    js = await score_one("deepseek-chat", CASE, DEFAULT_RUBRIC)
    assert js.per_criterion["relevance"] == 0.0
    assert js.per_criterion["correctness"] == 1.0


@respx.mock
async def test_score_case_fans_out_to_all_models():
    respx.post(DS_URL).mock(return_value=_resp(_score_obj(1.0), "deepseek-chat"))
    respx.post(QWEN_URL).mock(return_value=_resp(_score_obj(0.5), "qwen-plus"))
    scores = await score_case(CASE, ["deepseek-chat", "qwen-plus"], DEFAULT_RUBRIC)
    assert len(scores) == 2
    models = {s.model for s in scores}
    assert models == {"deepseek-chat", "qwen-plus"}
