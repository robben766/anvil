import json

import httpx
import respx
from anvil_code_agent.harness.loop import run
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _tool_call_resp(name, args):
    return httpx.Response(
        200,
        json={
            "id": "x", "model": "deepseek-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]},
                "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def _text_resp(text):
    return httpx.Response(
        200,
        json={"id": "x", "model": "deepseek-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )


def _reg():
    @tool(name="noop", description="d", params={}, required=[])
    def noop(args, ctx):
        return ToolResult(content="ok", ok=True)

    return ToolRegistry([noop])


@respx.mock
async def test_run_loops_until_done(tmp_path):
    # 第一次返回工具调用,第二次返回完成文本
    respx.post(DS_URL).mock(side_effect=[_tool_call_resp("noop", {}), _text_resp("done")])
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=10)
    final = await run(s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)))
    assert final.status == "done"
    assert final.step == 2


@respx.mock
async def test_run_guards_on_max_steps(tmp_path):
    # 永远返回工具调用 → 必须被 max_steps 截断
    respx.post(DS_URL).mock(return_value=_tool_call_resp("noop", {}))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=3)
    final = await run(s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)))
    assert final.status == "exhausted"
    assert final.step == 3
