import pytest
from anvil_eval.metrics.context_recall import context_recall
from anvil_eval.metrics.faithfulness import faithfulness


@pytest.fixture
def fake_judge(monkeypatch):
    """打桩 judge_json:第一类调用拆主张,第二类调用逐条判定支持。"""
    calls = {"split": [], "verify": []}

    def install(claims: list[str], verdicts: list[bool]):
        verdict_iter = iter(verdicts)

        async def _judge(instruction, payload):
            if "拆分" in instruction:
                calls["split"].append(payload)
                return {"reason": "r", "claims": claims}
            calls["verify"].append(payload)
            return {"reason": "r", "supported": next(verdict_iter)}

        monkeypatch.setattr("anvil_eval.metrics.claims.judge_json", _judge)

    install.calls = calls
    return install


async def test_faithfulness_hand_calculated(fake_judge):
    # 手算:3 条主张,支持 2 条 → 2/3
    fake_judge(["成立于1998", "总部杭州", "员工5万"], [True, True, False])
    score = await faithfulness(answer="…", contexts=["…"])
    assert score == pytest.approx(2 / 3)


async def test_faithfulness_no_claims_scores_zero(fake_judge):
    fake_judge([], [])
    assert await faithfulness(answer="嗯。", contexts=["c"]) == 0.0


async def test_context_recall_hand_calculated(fake_judge):
    # 手算:参考答案 4 条主张,context 覆盖 3 条 → 0.75
    fake_judge(["a", "b", "c", "d"], [True, True, True, False])
    score = await context_recall(reference="…", contexts=["…"])
    assert score == pytest.approx(0.75)
