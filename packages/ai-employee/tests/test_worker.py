import json

import httpx
import pytest
import respx
from anvil_ai_employee.scheduler.queue import enqueue
from anvil_ai_employee.worker import run_once

pytestmark = pytest.mark.asyncio

CHAT_URL = "https://api.deepseek.com/v1/chat/completions"  # MUST match anvil_gateway base


def _assistant_tool_call(tool_id, name, arguments):
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
                                "id": tool_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _assistant_done():
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "完成"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


@respx.mock
async def test_run_once_executes_job_to_done(engine, session_factory, monkeypatch):
    # ensure kb tables exist via engine fixture; seed a doc so kb_recent has data
    from datetime import UTC, datetime

    from anvil_kb.db import DocumentRow

    async with session_factory() as s:
        async with s.begin():
            s.add(
                DocumentRow(
                    title="新政策",
                    source_name="p.md",
                    content="保单新增条款……",
                    created_at=datetime(2026, 6, 8, tzinfo=UTC),
                )
            )

    route = respx.post(CHAT_URL)
    route.side_effect = [
        _assistant_tool_call("c1", "recall_marker", {}),
        _assistant_tool_call("c2", "kb_recent", {"since_iso": "2026-06-01T00:00:00+00:00"}),
        _assistant_tool_call(
            "c3",
            "submit_report",
            {
                "markdown": "# 周报\n- 新政策 (p.md)",
                "covered_until_iso": "2026-06-08T00:00:00+00:00",
            },
        ),
        _assistant_done(),
    ]

    job_id = await enqueue(session_factory, skill="kb_digest", payload={})
    ok = await run_once(session_factory, model="deepseek-chat", worker_id="w1", max_steps=12)
    assert ok is True

    from anvil_ai_employee.db import JobRow
    from sqlalchemy import select

    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one()
        assert row.status == "done"
        assert "周报" in row.result


@respx.mock
async def test_run_once_no_job_returns_false(engine, session_factory):
    assert await run_once(session_factory, model="deepseek-chat", worker_id="w1") is False


@respx.mock
async def test_worker_runs_researcher_employee(engine, session_factory):
    """A job tagged employee=researcher must run the researcher persona, not kb_reporter.
    We assert the agent was driven with the researcher's system prompt by inspecting the
    request body sent to the gateway."""
    captured = {}

    def _capture(request):
        body = json.loads(request.content)
        captured.setdefault("system", body["messages"][0]["content"])
        return _assistant_tool_call(
            "c1",
            "submit_report",
            {"markdown": "调研纪要", "covered_until_iso": "2026-06-09T00:00:00"},
        )

    respx.post(CHAT_URL).mock(side_effect=_capture)

    await enqueue(
        session_factory,
        skill="kb_digest",
        payload={"task": "调研向量检索"},
        employee="researcher",
    )
    ran = await run_once(session_factory, model="deepseek-chat", worker_id="w1")
    assert ran is True
    assert "调研员" in captured["system"]  # researcher persona, not kb_reporter
