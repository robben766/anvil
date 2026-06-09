import json

import httpx
import respx
from anvil_code_agent.harness.loop import run, step
from anvil_code_agent.harness.permission import deny_high
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _bash_call(cmd):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": json.dumps({"cmd": cmd})}}]},
         "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _text(t):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": t}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _reg():
    @tool(name="bash", description="d", params={"cmd": {"type": "string"}}, required=["cmd"])
    def bash(args, ctx):
        return ToolResult(content="ran:" + args["cmd"], ok=True)

    return ToolRegistry([bash])


@respx.mock
async def test_policy_blocks_high_risk_tool_as_feedback(tmp_path):
    respx.post(DS_URL).mock(return_value=_bash_call("echo hi"))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    # bash 是 high 风险;deny_high 应拦截 → 工具结果是"denied"反馈,不执行
    s2 = await step(
        s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)), policy=deny_high
    )
    tool_msg = s2.messages[-1]
    assert tool_msg["role"] == "tool"
    assert "denied" in tool_msg["content"].lower()
    assert "ran:" not in tool_msg["content"]  # 没真执行


@respx.mock
async def test_run_passes_policy_and_budget_through(tmp_path):
    respx.post(DS_URL).mock(side_effect=[_bash_call("ls"), _text("done")])
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    final = await run(s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)),
                      policy=deny_high, token_budget=10_000)
    assert final.status == "done"
    # bash 被拦,但循环没崩,继续到 done
    assert any("denied" in m.get("content", "") for m in final.messages if m["role"] == "tool")
