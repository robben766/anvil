import json

import httpx
import respx
from anvil_code_agent.harness.loop import step
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _tool_call_resp(name: str, args: dict):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _text_resp(text: str):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _echo_registry():
    @tool(name="echo", description="d", params={"text": {"type": "string"}}, required=["text"])
    def echo(args, ctx):
        return ToolResult(content="echoed:" + args["text"], ok=True)

    return ToolRegistry([echo])


@respx.mock
async def test_step_executes_tool_call_and_appends_messages(tmp_path):
    respx.post(DS_URL).mock(return_value=_tool_call_resp("echo", {"text": "hi"}))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    s2 = await step(s, "deepseek-chat", _echo_registry(), ToolContext(workdir=str(tmp_path)))
    # 追加了 assistant(带 tool_calls) + tool 结果 两条消息,步进 +1,仍 running
    roles = [m["role"] for m in s2.messages]
    assert roles[-2:] == ["assistant", "tool"]
    assert s2.messages[-1]["tool_call_id"] == "call_1"
    assert "echoed:hi" in s2.messages[-1]["content"]
    assert s2.step == 1 and s2.status == "running"


@respx.mock
async def test_step_finishes_on_text_response(tmp_path):
    respx.post(DS_URL).mock(return_value=_text_resp("all done"))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    s2 = await step(s, "deepseek-chat", _echo_registry(), ToolContext(workdir=str(tmp_path)))
    assert s2.status == "done"
    assert s2.messages[-1]["content"] == "all done"
