import sys
from datetime import UTC, datetime

import pytest
from anvil_ai_employee.cli import add_schedule, run_now, show_report

pytestmark = pytest.mark.asyncio


async def test_add_schedule_inserts_with_next_run(session_factory):
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    sid = await add_schedule(session_factory, name="周报", cron="0 9 * * 1",
                             skill="kb_digest", now=now)
    from anvil_ai_employee.db import ScheduleRow
    from sqlalchemy import select
    async with session_factory() as s:
        row = (await s.execute(select(ScheduleRow).where(ScheduleRow.id == sid))).scalar_one()
        assert row.name == "周报" and row.next_run_at > now


async def test_run_now_enqueues(session_factory):
    jid = await run_now(session_factory, skill="kb_digest")
    from anvil_ai_employee.db import JobRow
    from sqlalchemy import select
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == jid))).scalar_one()
        assert row.status == "pending" and row.skill == "kb_digest"


async def test_show_report_for_missing_job(session_factory):
    import uuid
    out = await show_report(session_factory, job_id=uuid.uuid4())
    assert "未找到" in out or "not found" in out.lower()


async def test_make_strategy_mem0_and_none(session_factory):
    from anvil_ai_employee.cli import make_strategy
    from anvil_ai_employee.memory.mem0 import Mem0Strategy
    from anvil_ai_employee.memory.strategy import NoMemoryStrategy
    assert isinstance(make_strategy("none", session_factory, "deepseek-chat"), NoMemoryStrategy)
    assert isinstance(make_strategy("mem0", session_factory, "deepseek-chat"), Mem0Strategy)


async def test_make_strategy_letta(session_factory):
    from anvil_ai_employee.cli import make_strategy
    from anvil_ai_employee.memory.letta import LettaStrategy
    assert isinstance(make_strategy("letta", session_factory, "deepseek-chat"), LettaStrategy)


async def test_inbox_list_and_resolve_helpers(session_factory):
    # 用 InboxStore 直接造一个 pending,测 cli 的 inbox_list_text / inbox_resolve 纯函数
    import json

    from anvil_ai_employee.cli import inbox_list_text, inbox_resolve
    from anvil_ai_employee.inbox import InboxStore
    from anvil_code_agent.state import AgentState
    msgs = ({"role": "system", "content": "s"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": json.dumps({"cmd": "rm"})}}]})
    st = AgentState(messages=msgs, step=1, max_steps=10, workdir="/tmp", status="suspended")
    iid = await InboxStore(session_factory).suspend(employee="assistant", state=st)
    txt = await inbox_list_text(session_factory)
    assert "bash" in txt and str(iid)[:8] in txt
    await inbox_resolve(session_factory, inbox_id=iid, decision="reject", payload={"reason": "no"})
    after = await inbox_list_text(session_factory)
    assert "(空)" in after or "pending" not in after.lower()


_MCP_SERVER = ["-m", "anvil_ai_employee.mcp.mock_servers.email_server"]


async def test_mcp_list_tools_text(session_factory):
    from anvil_ai_employee.cli import mcp_list_tools_text
    from anvil_ai_employee.mcp.connector import ConnectorConfig

    cfg = ConnectorConfig(name="gmail", command=sys.executable, args=_MCP_SERVER)
    text = await mcp_list_tools_text(session_factory, employee="alice", config=cfg)
    assert "gmail__list_events" in text
    assert "low" in text and "high" in text


async def test_mcp_put_token(session_factory):
    from anvil_ai_employee.cli import mcp_put_token
    from anvil_ai_employee.mcp.tokens import McpTokenStore

    await mcp_put_token(
        session_factory, employee="alice", connector="gmail", env_key="GMAIL_TOKEN", secret="T"
    )
    env = await McpTokenStore(session_factory).env_for(employee="alice", connector="gmail")
    assert env == {"GMAIL_TOKEN": "T"}


async def test_team_run_creates_goal_and_children(session_factory, respx_mock):
    import json
    import uuid

    import httpx
    from anvil_ai_employee.cli import team_run
    from anvil_ai_employee.db import GoalRow, JobRow
    from sqlalchemy import select

    plan = {
        "subtasks": [
            {"employee": "researcher", "task": "调研 X"},
            {"employee": "kb_reporter", "task": "写周报"},
        ]
    }
    respx_mock.route(method="POST", url__regex=r".*chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": json.dumps(plan)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    gid = await team_run(session_factory, objective="调研 X 并写周报", model="deepseek-chat")
    assert isinstance(gid, uuid.UUID)
    async with session_factory() as s:
        goal = (await s.execute(select(GoalRow).where(GoalRow.id == gid))).scalar_one()
        assert goal.status == "running"
        children = (await s.execute(select(JobRow).where(JobRow.goal_id == gid))).scalars().all()
        assert {c.employee for c in children} == {"researcher", "kb_reporter"}


async def test_team_status_text_lists_children(session_factory):
    import uuid

    from anvil_ai_employee.cli import team_status_text
    from anvil_ai_employee.db import GoalRow, JobRow

    gid = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            s.add(GoalRow(id=gid, objective="g", status="running"))
            s.add(JobRow(
                skill="kb_digest", payload={"task": "a"}, status="done",
                result="r", goal_id=gid, employee="researcher",
            ))
            s.add(JobRow(
                skill="kb_digest", payload={"task": "b"}, status="pending",
                goal_id=gid, employee="kb_reporter",
            ))
    text = await team_status_text(session_factory, goal_id=gid, model="deepseek-chat")
    assert "researcher" in text and "kb_reporter" in text
    assert "done" in text and "pending" in text
    # not all terminal → no synthesis yet
    assert "最终产出" not in text
