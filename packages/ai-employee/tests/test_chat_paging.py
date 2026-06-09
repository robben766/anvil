import pytest
from anvil_ai_employee.chat import apply_self_paging
from anvil_code_agent.harness.context import estimate_tokens

pytestmark = pytest.mark.asyncio


def test_apply_self_paging_truncates_when_over_budget():
    # build an over-budget history of tool messages
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for _ in range(40):
        msgs.append({"role": "assistant", "content": "x" * 400})
        msgs.append({"role": "user", "content": "y" * 400})
    before = estimate_tokens(msgs)
    out, warned = apply_self_paging(msgs, budget=before // 2, warn_ratio=0.7, summarizer=None)
    assert estimate_tokens(out) <= before  # shrunk or equal
    assert out[0]["role"] == "system"  # system protected


def test_apply_self_paging_warns_near_budget():
    msgs = [{"role": "system", "content": "s" * 100}, {"role": "user", "content": "u" * 100}]
    # tokens ~ 50; budget 60, 0.7*60=42 → over warn but under flush
    out, warned = apply_self_paging(msgs, budget=60, warn_ratio=0.7, summarizer=None)
    assert warned is True
    assert any("上下文" in (m.get("content") or "") for m in out if m["role"] == "system")
