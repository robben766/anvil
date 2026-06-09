import pytest
from anvil_ai_employee.inbox import InboxStore
from anvil_code_agent.state import AgentState

pytestmark = pytest.mark.asyncio


def _suspended():
    import json
    msgs = ({"role": "system", "content": "s"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": json.dumps({"cmd": "rm -rf /"})}}]})
    return AgentState(messages=msgs, step=1, max_steps=10, workdir="/tmp", status="suspended")


async def test_suspend_then_list_and_resolve(session_factory):
    store = InboxStore(session_factory)
    iid = await store.suspend(employee="assistant", state=_suspended())
    pend = await store.list_pending()
    assert len(pend) == 1 and pend[0].tool_name == "bash" and pend[0].risk == "high"
    assert pend[0].tool_args == {"cmd": "rm -rf /"}
    await store.resolve(iid, decision="reject", payload={"reason": "no"})
    row = await store.get(iid)
    assert row.status == "resolved" and row.decision == "reject"
    assert await store.list_pending() == []


async def test_state_roundtrips(session_factory):
    store = InboxStore(session_factory)
    iid = await store.suspend(employee="assistant", state=_suspended())
    row = await store.get(iid)
    from anvil_code_agent.harness.recovery import load_state
    st = load_state(row.state_json)
    assert st.status == "suspended" and len(st.messages) == 3
