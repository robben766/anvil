# packages/ai-employee/tests/test_supervisor.py
import json

import pytest

pytestmark = pytest.mark.asyncio


async def test_decompose_parses_subtasks(respx_mock):
    import httpx
    from anvil_ai_employee.fleet.supervisor import decompose

    payload = {
        "subtasks": [
            {"employee": "researcher", "task": "调研向量检索"},
            {"employee": "kb_reporter", "task": "产周报"},
        ]
    }
    respx_mock.route(method="POST", url__regex=r".*chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    subs = await decompose(
        "调研并产周报", model="deepseek-chat", employees=["researcher", "kb_reporter"]
    )
    assert [(s.employee, s.task) for s in subs] == [
        ("researcher", "调研向量检索"),
        ("kb_reporter", "产周报"),
    ]


async def test_decompose_filters_unknown_employee(respx_mock):
    import httpx
    from anvil_ai_employee.fleet.supervisor import decompose

    payload = {
        "subtasks": [
            {"employee": "ghost", "task": "x"},
            {"employee": "researcher", "task": "ok"},
        ]
    }
    respx_mock.route(method="POST", url__regex=r".*chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    subs = await decompose("g", model="deepseek-chat", employees=["researcher"])
    assert [s.employee for s in subs] == ["researcher"]


async def test_decompose_empty_falls_back_to_single(respx_mock):
    import httpx
    from anvil_ai_employee.fleet.supervisor import decompose

    respx_mock.route(method="POST", url__regex=r".*chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps({"subtasks": []})}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    subs = await decompose("解决世界饥饿", model="deepseek-chat", employees=["researcher"])
    assert len(subs) == 1
    assert subs[0].employee == "researcher"
    assert subs[0].task == "解决世界饥饿"


async def test_fan_out_enqueues_child_jobs(session_factory):
    import uuid

    from anvil_ai_employee.db import JobRow
    from anvil_ai_employee.fleet.supervisor import SubTask, fan_out
    from sqlalchemy import select

    gid = uuid.uuid4()
    ids = await fan_out(
        session_factory,
        goal_id=gid,
        subtasks=[
            SubTask(employee="researcher", task="a"),
            SubTask(employee="kb_reporter", task="b"),
        ],
    )
    assert len(ids) == 2
    async with session_factory() as s:
        rows = (await s.execute(select(JobRow).where(JobRow.goal_id == gid))).scalars().all()
        assert {r.employee for r in rows} == {"researcher", "kb_reporter"}
        assert {r.payload["task"] for r in rows} == {"a", "b"}
