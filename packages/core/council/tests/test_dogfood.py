import json
import os
from pathlib import Path

import httpx
import pytest
import respx
from anvil_council.aggregate import aggregate
from anvil_council.seats import score_case

DS_URL = "https://api.deepseek.com/v1/chat/completions"
DOGFOOD = Path(__file__).resolve().parents[1] / "golden" / "kb_answers.jsonl"


def _resp(score: float):
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
            "model": "deepseek-chat",
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
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )


def test_dogfood_dataset_loads():
    rows = [json.loads(x) for x in DOGFOOD.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) >= 6
    for r in rows:
        assert r["question"] and r["reference"] and r["answer"]


@respx.mock
async def test_jury_scores_a_dogfood_case():
    respx.post(DS_URL).mock(return_value=_resp(1.0))
    rows = [json.loads(x) for x in DOGFOOD.read_text(encoding="utf-8").splitlines() if x.strip()]
    scores = await score_case(rows[0], ["deepseek-chat"])
    v = aggregate(scores)
    assert 0.0 <= v.overall <= 1.0
    assert v.confidence == 1.0
