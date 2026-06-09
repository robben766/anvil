import sys

import pytest
from anvil_ai_employee.hitl import HitlDecision

_SERVER = ["-m", "anvil_ai_employee.mcp.mock_servers.email_server"]


def test_mcp_risk_policy_routes_by_tool_risk():
    from anvil_ai_employee.mcp.connector import mcp_risk_policy

    policy = mcp_risk_policy({"gmail__send_email": "high", "gmail__list_events": "low"})
    assert policy("gmail__send_email", {}, "high") == HitlDecision.SUSPEND
    assert policy("gmail__list_events", {}, "low") == HitlDecision.EXECUTE
    # unknown tool falls back to risk_level() (unknown → high → suspend)
    assert policy("mystery_tool", {}, "high") == HitlDecision.SUSPEND


@pytest.mark.asyncio
async def test_build_mcp_registry_lists_tools_and_policy(session_factory):
    from anvil_ai_employee.mcp.connector import ConnectorConfig, build_mcp_registry
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    store = McpTokenStore(session_factory)
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="TOK")
    cfg = ConnectorConfig(name="gmail", command=sys.executable, args=_SERVER)
    registry, clients, policy = await build_mcp_registry(
        configs=[cfg], employee="alice", token_store=store
    )
    try:
        names = {s["function"]["name"] for s in registry.schemas()}
        assert names == {"gmail__list_events", "gmail__send_email"}
        assert policy("gmail__list_events", {}, "low") == HitlDecision.EXECUTE
        assert policy("gmail__send_email", {}, "high") == HitlDecision.SUSPEND
    finally:
        for c in clients:
            c.close()


@pytest.mark.asyncio
async def test_hitl_step_executes_read_suspends_write(session_factory):
    """End-to-end through the real mock server: read tool runs, write tool suspends."""
    from anvil_ai_employee.hitl import hitl_step
    from anvil_ai_employee.mcp.connector import ConnectorConfig, build_mcp_registry
    from anvil_ai_employee.mcp.tokens import McpTokenStore
    from anvil_code_agent.state import AgentState
    from anvil_code_agent.tools.base import ToolContext

    store = McpTokenStore(session_factory)
    await store.put(employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="TOK")
    cfg = ConnectorConfig(name="gmail", command=sys.executable, args=_SERVER)
    registry, clients, policy = await build_mcp_registry(
        configs=[cfg], employee="alice", token_store=store
    )
    ctx = ToolContext(workdir="/tmp")
    try:
        # craft a state whose last assistant message proposes a read-only tool call
        def state_calling(tool_name, args):
            st = AgentState.new(system="p", task="t", workdir="/tmp", max_steps=5)
            fn = {"name": tool_name, "arguments": __import__("json").dumps(args)}
            assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": fn}],
            }
            return st.append(assistant)

        read_state = state_calling("gmail__list_events", {"date": "2026-06-09"})
        after_read = await hitl_step(read_state, "deepseek-chat", registry, ctx, policy=policy)
        assert after_read.status == "running"
        assert any(
            m.get("role") == "tool" and "standup" in (m.get("content") or "")
            for m in after_read.messages
        )

        write_state = state_calling("gmail__send_email", {"to": "x", "subject": "s", "body": "b"})
        after_write = await hitl_step(write_state, "deepseek-chat", registry, ctx, policy=policy)
        assert after_write.status == "suspended"
    finally:
        for c in clients:
            c.close()
