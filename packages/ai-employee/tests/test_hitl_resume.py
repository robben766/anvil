import json

import httpx
import pytest
import respx
from anvil_ai_employee.hitl import HitlDecision, apply_decision, hitl_run
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"
TC = ToolContext(workdir="/tmp")


def _registry(calls):
    @tool(name="danger", description="d", params={"t": {"type": "string"}}, required=["t"])
    def danger(args, ctx):
        calls.append(args["t"])
        return ToolResult(content="DID:" + args["t"], ok=True)
    return ToolRegistry([danger])


def _suspended_state():
    # an assistant message proposing danger(t=rm), unanswered → suspended
    msgs = (
        {"role": "system", "content": "s"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "danger", "arguments": json.dumps({"t": "rm"})}}]},
    )
    return AgentState(messages=msgs, step=1, max_steps=10, workdir="/tmp", status="suspended")


async def test_approve_executes_original_args():
    calls = []
    st = apply_decision(_suspended_state(), decision="approve", payload={},
                        registry=_registry(calls), ctx=TC)
    assert st.status == "running"
    assert calls == ["rm"]  # executed with original args
    assert any(m["role"] == "tool" and "DID:rm" in m["content"] for m in st.messages)


async def test_edit_executes_new_args():
    calls = []
    apply_decision(_suspended_state(), decision="edit", payload={"args": {"t": "safe"}},
                   registry=_registry(calls), ctx=TC)
    assert calls == ["safe"]


async def test_reject_injects_feedback_no_exec():
    calls = []
    st = apply_decision(_suspended_state(), decision="reject", payload={"reason": "太危险"},
                        registry=_registry(calls), ctx=TC)
    assert calls == []
    assert any(m["role"] == "tool" and "太危险" in m["content"] for m in st.messages)
    assert st.status == "running"


async def test_respond_injects_custom_no_exec():
    calls = []
    st = apply_decision(_suspended_state(), decision="respond", payload={"message": "我帮你做了"},
                        registry=_registry(calls), ctx=TC)
    assert calls == []
    assert any(m["role"] == "tool" and "我帮你做了" in m["content"] for m in st.messages)


@respx.mock
async def test_resume_continues_to_done():
    # after approve, model wraps up
    respx.post(DS_URL).mock(return_value=httpx.Response(200, json={
        "id": "x", "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "完成"},
        "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    calls = []
    reg = _registry(calls)
    st = apply_decision(_suspended_state(), decision="approve", payload={}, registry=reg, ctx=TC)
    out = await hitl_run(st, "deepseek-chat", reg, TC, policy=lambda n, a, r: HitlDecision.EXECUTE)
    assert out.status == "done"
