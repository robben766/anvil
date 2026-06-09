import json

import httpx
import pytest
import respx
from anvil_ai_employee.hitl import HitlDecision
from anvil_ai_employee.inbox import InboxStore
from anvil_ai_employee.inbox_resume import resume_from_inbox
from anvil_ai_employee.memory.store import MemoryStore
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"
TC = ToolContext(workdir="/tmp")


class StubEmbedder:
    def embed_texts(self, texts):
        return [[0.5] * 512 for _ in texts]

    def embed_query(self, text):
        return [0.5] * 512


def _registry(calls):
    @tool(name="danger", description="d", params={"t": {"type": "string"}}, required=["t"])
    def danger(args, ctx):
        calls.append(args["t"])
        return ToolResult(content="DID:" + args["t"], ok=True)
    return ToolRegistry([danger])


def _text(t):
    return httpx.Response(200, json={
        "id": "x", "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": t},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


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


@respx.mock
async def test_resume_from_inbox_closes_the_loop(session_factory):
    # after approve, model wraps up → done
    respx.post(DS_URL).mock(side_effect=[_text("完成")])
    calls = []
    reg = _registry(calls)
    store = InboxStore(session_factory)

    # suspend → resolve(approve) → resume
    iid = await store.suspend(employee="assistant", state=_suspended_state())
    pending = await store.list_pending()
    assert len(pending) == 1
    await store.resolve(iid, decision="approve", payload={})
    inbox_row = await store.get(iid)
    assert inbox_row.status == "resolved" and inbox_row.decision == "approve"

    out = await resume_from_inbox(
        inbox_row, registry=reg, ctx=TC, model="deepseek-chat",
        session_factory=session_factory, embedder=StubEmbedder(),
        policy=lambda n, a, r: HitlDecision.EXECUTE)

    # ran to completion
    assert out.status == "done"
    # danger tool executed with original (approved) args
    assert calls == ["rm"]
    assert any(m["role"] == "tool" and "DID:rm" in m["content"] for m in out.messages)
    # intervention recorded to long-term memory
    facts = await MemoryStore(session_factory).list_facts(employee="assistant", kind="hitl")
    assert len(facts) == 1 and "danger" in facts[0].content
