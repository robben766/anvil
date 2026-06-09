import uuid

import pytest

pytestmark = pytest.mark.asyncio


async def _seed_goal_with_children(session_factory, child_statuses):
    """Insert a GoalRow + one JobRow per (status, result) tuple. Returns goal_id."""
    from anvil_ai_employee.db import GoalRow, JobRow

    gid = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            s.add(GoalRow(id=gid, objective="调研并产周报", status="running"))
            for i, (status, result) in enumerate(child_statuses):
                s.add(
                    JobRow(
                        skill="kb_digest",
                        payload={"task": f"t{i}"},
                        status=status,
                        result=result,
                        goal_id=gid,
                        employee="researcher",
                    )
                )
    return gid


async def test_children_terminal_false_when_pending(session_factory):
    from anvil_ai_employee.fleet.aggregator import children_terminal

    gid = await _seed_goal_with_children(session_factory, [("done", "r"), ("pending", None)])
    assert await children_terminal(session_factory, gid) is False


async def test_children_terminal_true_when_all_done_or_failed(session_factory):
    from anvil_ai_employee.fleet.aggregator import children_terminal

    gid = await _seed_goal_with_children(session_factory, [("done", "r"), ("failed", None)])
    assert await children_terminal(session_factory, gid) is True


async def test_aggregate_returns_none_when_not_terminal(session_factory):
    from anvil_ai_employee.fleet.aggregator import aggregate

    gid = await _seed_goal_with_children(session_factory, [("done", "r"), ("running", None)])
    assert await aggregate(session_factory, gid, model="deepseek-chat") is None


async def test_aggregate_synthesizes_and_writes_result(session_factory, respx_mock):
    import httpx
    from anvil_ai_employee.db import GoalRow
    from anvil_ai_employee.fleet.aggregator import aggregate
    from sqlalchemy import select

    gid = await _seed_goal_with_children(
        session_factory, [("done", "调研结论 A"), ("failed", None)]
    )
    respx_mock.route(method="POST", url__regex=r".*chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "最终综合交付物"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    out = await aggregate(session_factory, gid, model="deepseek-chat")
    assert out == "最终综合交付物"
    async with session_factory() as s:
        goal = (await s.execute(select(GoalRow).where(GoalRow.id == gid))).scalar_one()
        assert goal.status == "done"
        assert goal.result == "最终综合交付物"
