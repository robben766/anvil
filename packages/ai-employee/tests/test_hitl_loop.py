import json

import httpx
import pytest
import respx
from anvil_ai_employee.hitl import HitlDecision, _unanswered_tool_calls, hitl_run
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"
TC = ToolContext(workdir="/tmp")


def _tool_call(tcid, name, args):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": tcid, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]},
         "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _text(t):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": t}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _registry():
    @tool(name="safe_echo", description="d", params={"t": {"type": "string"}}, required=["t"])
    def safe_echo(args, ctx):
        return ToolResult(content="echo:" + args["t"], ok=True)

    @tool(name="danger", description="d", params={"t": {"type": "string"}}, required=["t"])
    def danger(args, ctx):
        return ToolResult(content="DID:" + args["t"], ok=True)
    return ToolRegistry([safe_echo, danger])


def _policy(name, args, risk):
    # danger is unknown→high→suspend; safe_echo unknown→high too — force safe low here
    return HitlDecision.SUSPEND if name == "danger" else HitlDecision.EXECUTE


@respx.mock
async def test_runs_safe_tool_to_done():
    respx.post(DS_URL).mock(side_effect=[_tool_call("c1", "safe_echo", {"t": "hi"}), _text("done")])
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=10)
    out = await hitl_run(s, "deepseek-chat", _registry(), TC, policy=_policy)
    assert out.status == "done"
    assert any(m.get("content") == "echo:hi" for m in out.messages if m["role"] == "tool")


@respx.mock
async def test_suspends_on_high_risk():
    respx.post(DS_URL).mock(side_effect=[_tool_call("c1", "danger", {"t": "rm"})])
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=10)
    out = await hitl_run(s, "deepseek-chat", _registry(), TC, policy=_policy)
    assert out.status == "suspended"
    pending = _unanswered_tool_calls(out)
    assert len(pending) == 1 and pending[0]["function"]["name"] == "danger"
    # the danger tool was NOT executed (no tool message for it)
    assert not any(m["role"] == "tool" for m in out.messages)
