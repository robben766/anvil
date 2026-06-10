# P4-M5 Multi-Employee Fleet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the single-employee pipeline into a team: a supervisor decomposes a goal into independent subtasks, fans them out onto the M1 PG queue tagged per-employee, a worker pool runs them in parallel (reusing M1's SKIP-LOCKED claim), and an aggregator synthesizes the final result once all child jobs are terminal.

**Architecture:** New `fleet/` subpackage in `anvil_ai_employee`: `team.py` (an `Employee` registry — `kb_reporter` + a demo `researcher`), `supervisor.py` (`decompose` via `guard.structured_chat` + `fan_out` via M1 `enqueue`), `aggregator.py` (`children_terminal` + `aggregate`). Storage adds an `ae_goals` table and two nullable columns on `ae_jobs` (`goal_id`, `employee`). `worker.run_once` is generalized to pick persona/registry by `job.employee` (falling back to M1's `kb_reporter`/skill path). Reuses M1 queue/worker + P3 harness + `guard.structured_chat`; M2/M3/M4 employees are unchanged.

**Tech Stack:** Python 3.12, SQLAlchemy 2 + asyncpg + PG@5434, `anvil-guard` structured_chat, pytest/pytest-asyncio + respx. Reuses `anvil_ai_employee.scheduler.queue`, `worker`, `tools`, `skills.kb_digest`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | Declare `anvil-guard` dependency (already imported transitively by mem0; make it explicit) |
| `src/anvil_ai_employee/db.py` (modify) | `GoalRow` (`ae_goals`) + `JobRow.goal_id` / `JobRow.employee` columns |
| `src/anvil_ai_employee/scheduler/queue.py` (modify) | `enqueue` gains `goal_id`/`employee`; `ClaimedJob` gains `goal_id`/`employee`; `claim_one` populates them |
| `src/anvil_ai_employee/skills/research.py` (create) | `researcher` persona + registry (reuses `build_employee_tools`) |
| `src/anvil_ai_employee/fleet/__init__.py` (create) | Package marker |
| `src/anvil_ai_employee/fleet/team.py` (create) | `Employee` dataclass + `EMPLOYEES` registry |
| `src/anvil_ai_employee/fleet/supervisor.py` (create) | `SubTask`, `decompose`, `fan_out` |
| `src/anvil_ai_employee/fleet/aggregator.py` (create) | `children_terminal`, `aggregate` |
| `src/anvil_ai_employee/worker.py` (modify) | Select employee by `job.employee`; task from `job.payload['task']` |
| `src/anvil_ai_employee/cli.py` (modify) | `team run` / `team status` subcommands |
| `examples/12-ai-employee-fleet/README.md` (create) | Milestone walkthrough |
| `CLAUDE.md` (modify) | M5 subsection |

**Test files:** `tests/test_ai_employee_db.py` (extend), `tests/test_queue.py` (extend), `tests/test_team.py`, `tests/test_supervisor.py`, `tests/test_aggregator.py`, `tests/test_worker.py` (extend), `tests/test_ai_employee_cli.py` (extend).

**Verification rule (M1–M4 lesson):** after every task run `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`, and once across the repo root before the PR. Live tests are opt-in only.

---

### Task 1: `ae_goals` table + `JobRow` columns + explicit `anvil-guard` dep

**Files:**
- Modify: `packages/ai-employee/pyproject.toml`
- Modify: `packages/ai-employee/src/anvil_ai_employee/db.py`
- Test: `packages/ai-employee/tests/test_ai_employee_db.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `packages/ai-employee/tests/test_ai_employee_db.py`:

```python
async def test_goal_row_and_job_fleet_columns(session_factory):
    import uuid

    from sqlalchemy import select

    from anvil_ai_employee.db import GoalRow, JobRow

    gid = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            s.add(GoalRow(id=gid, objective="调研并产周报", status="running"))
            s.add(
                JobRow(
                    skill="kb_digest",
                    payload={"task": "查 X"},
                    goal_id=gid,
                    employee="researcher",
                )
            )
    async with session_factory() as s:
        goal = (await s.execute(select(GoalRow).where(GoalRow.id == gid))).scalar_one()
        assert goal.objective == "调研并产周报"
        assert goal.status == "running"
        assert goal.result is None
        job = (
            await s.execute(select(JobRow).where(JobRow.goal_id == gid))
        ).scalar_one()
        assert job.employee == "researcher"
        assert job.goal_id == gid


async def test_job_columns_default_null(session_factory):
    """M1 jobs created without goal_id/employee must still work (nullable)."""
    import uuid

    from sqlalchemy import select

    from anvil_ai_employee.db import JobRow

    jid = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            s.add(JobRow(id=jid, skill="kb_digest", payload={}))
    async with session_factory() as s:
        job = (await s.execute(select(JobRow).where(JobRow.id == jid))).scalar_one()
        assert job.goal_id is None
        assert job.employee is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_ai_employee_db.py -k "goal_row or fleet or default_null" -v`
Expected: FAIL — `cannot import name 'GoalRow'` / `JobRow` has no attribute `goal_id`.

- [ ] **Step 3: Add the column declarations to `JobRow` in `db.py`**

In `class JobRow`, after the `schedule_id` line, add:

```python
    goal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    employee: Mapped[str | None] = mapped_column(Text, nullable=True)
```

- [ ] **Step 4: Add the `GoalRow` table at the end of `db.py`**

```python
class GoalRow(Base):
    __tablename__ = "ae_goals"

    # A fleet goal: the supervisor decomposes `objective` into child jobs (ae_jobs.goal_id),
    # the aggregator synthesizes their results into `result` once all children are terminal.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 5: Declare `anvil-guard` explicitly in `pyproject.toml`**

In `packages/ai-employee/pyproject.toml`, add `"anvil-guard",` to the `dependencies` list (after `"anvil-kb",`), and add `anvil-guard = { workspace = true }` to the `[tool.uv.sources]` table (matching the other workspace entries). Then run `uv sync` from the repo root if needed.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_ai_employee_db.py -v`
Expected: all pass (new + existing).

- [ ] **Step 7: Ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`
Expected: ruff clean; all pass.

- [ ] **Step 8: Commit**

```bash
git add packages/ai-employee/pyproject.toml packages/ai-employee/src/anvil_ai_employee/db.py packages/ai-employee/tests/test_ai_employee_db.py
git -c user.email=robben766@users.noreply.github.com commit -m "feat(ai-employee): ae_goals table + JobRow goal_id/employee columns (fleet storage)"
```

---

### Task 2: `enqueue` / `ClaimedJob` / `claim_one` carry `goal_id` + `employee`

**Files:**
- Modify: `packages/ai-employee/src/anvil_ai_employee/scheduler/queue.py`
- Test: `packages/ai-employee/tests/test_queue.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `packages/ai-employee/tests/test_queue.py`:

```python
async def test_enqueue_and_claim_carry_goal_and_employee(session_factory):
    import uuid

    from anvil_ai_employee.scheduler.queue import claim_one, enqueue

    gid = uuid.uuid4()
    await enqueue(
        session_factory,
        skill="kb_digest",
        payload={"task": "查 X"},
        goal_id=gid,
        employee="researcher",
    )
    claimed = await claim_one(session_factory, worker_id="w1")
    assert claimed is not None
    assert claimed.goal_id == gid
    assert claimed.employee == "researcher"
    assert claimed.payload == {"task": "查 X"}


async def test_enqueue_without_goal_defaults_null(session_factory):
    from anvil_ai_employee.scheduler.queue import claim_one, enqueue

    await enqueue(session_factory, skill="kb_digest", payload={})
    claimed = await claim_one(session_factory, worker_id="w1")
    assert claimed is not None
    assert claimed.goal_id is None
    assert claimed.employee is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_queue.py -k "carry_goal or without_goal" -v`
Expected: FAIL — `enqueue() got an unexpected keyword argument 'goal_id'`.

- [ ] **Step 3: Extend `ClaimedJob`**

In `queue.py`, add two fields to the `ClaimedJob` dataclass (after `started_at`):

```python
    goal_id: uuid.UUID | None = None
    employee: str | None = None
```

- [ ] **Step 4: Extend `enqueue`**

Change the `enqueue` signature and insert. Replace the signature line block and the `insert(...).values(...)` block:

```python
async def enqueue(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    skill: str,
    payload: dict[str, Any],
    schedule_id: uuid.UUID | None = None,
    goal_id: uuid.UUID | None = None,
    employee: str | None = None,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                insert(JobRow).values(
                    id=job_id,
                    schedule_id=schedule_id,
                    skill=skill,
                    payload=payload,
                    status="pending",
                    goal_id=goal_id,
                    employee=employee,
                )
            )
    return job_id
```

- [ ] **Step 5: Populate them in `claim_one`**

In `claim_one`, in the `return ClaimedJob(...)` construction, add the two fields:

```python
            return ClaimedJob(
                id=fresh.id,
                skill=fresh.skill,
                payload=fresh.payload,
                status=fresh.status,
                locked_by=fresh.locked_by,
                started_at=fresh.started_at,
                goal_id=fresh.goal_id,
                employee=fresh.employee,
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_queue.py -v`
Expected: all pass (new + existing M1 queue tests).

- [ ] **Step 7: Ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`
Expected: ruff clean; all pass.

- [ ] **Step 8: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/scheduler/queue.py packages/ai-employee/tests/test_queue.py
git -c user.email=robben766@users.noreply.github.com commit -m "feat(ai-employee): queue enqueue/claim carry goal_id + employee"
```

---

### Task 3: `researcher` skill + `Employee` registry (`fleet/team.py`)

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/skills/research.py`
- Create: `packages/ai-employee/src/anvil_ai_employee/fleet/__init__.py`
- Create: `packages/ai-employee/src/anvil_ai_employee/fleet/team.py`
- Test: `packages/ai-employee/tests/test_team.py`

The `researcher` reuses `build_employee_tools` (so it has `kb_search` + `submit_report` to deliver a memo) under `employee="researcher"`; it differs from `kb_reporter` by persona/role. Deeper tool heterogeneity (e.g. an MCP employee) is a spiral.

- [ ] **Step 1: Write the failing test**

```python
# packages/ai-employee/tests/test_team.py
def test_employees_registry_has_two_distinct_employees():
    from anvil_ai_employee.fleet.team import EMPLOYEES

    assert set(EMPLOYEES) >= {"kb_reporter", "researcher"}
    kb = EMPLOYEES["kb_reporter"]
    rs = EMPLOYEES["researcher"]
    assert kb.name == "kb_reporter"
    assert rs.name == "researcher"
    assert kb.persona != rs.persona  # distinct roles
    assert kb.description and rs.description  # supervisor needs capability blurbs


def test_employee_build_registry_returns_toolregistry(session_factory):
    from anvil_code_agent.tools.base import ToolRegistry

    from anvil_ai_employee.fleet.team import EMPLOYEES
    from anvil_ai_employee.tools import EmployeeContext

    ctx = EmployeeContext(session_factory=session_factory, employee="researcher", job_id=None)
    reg = EMPLOYEES["researcher"].build_registry(ctx)
    assert isinstance(reg, ToolRegistry)
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "kb_search" in names
    assert "submit_report" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_team.py -v`
Expected: FAIL — `No module named 'anvil_ai_employee.fleet'`.

- [ ] **Step 3: Create the `researcher` skill**

```python
# packages/ai-employee/src/anvil_ai_employee/skills/research.py
"""Researcher skill: a fleet teammate that investigates a topic against the KB and
delivers a research memo. Reuses the shared employee tool library (kb_search to dig,
submit_report to deliver); differs from kb_reporter by role/persona, not tools."""

from __future__ import annotations

from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.tools import EmployeeContext, build_employee_tools

PERSONA = """你是「调研员」。你的任务是针对被指派的主题,在知识库里做深入调研,产出一份结构化中文调研纪要。

严格按步骤:
1. 调 `kb_search(query=<你要调研的主题或子问题>)` 检索相关片段,可多次换不同查询深挖。
2. 综合检索到的片段,写一份结构化中文调研纪要:要点分条、每条标注来源 source、指出证据强弱与缺口。
3. 调 `submit_report(markdown=<你的调研纪要>, covered_until_iso=<当前任务时间 ISO>)` 提交。
提交后即完成,不要再调用其他工具。"""


def build_registry(ctx: EmployeeContext) -> ToolRegistry:
    return ToolRegistry(build_employee_tools(ctx))
```

- [ ] **Step 4: Create the `fleet` package marker**

```python
# packages/ai-employee/src/anvil_ai_employee/fleet/__init__.py
"""Multi-employee fleet: a supervisor decomposes a goal into subtasks, fans them out
onto the M1 PG queue per-employee, and an aggregator synthesizes the final result."""
```

- [ ] **Step 5: Create the `Employee` registry**

```python
# packages/ai-employee/src/anvil_ai_employee/fleet/team.py
"""The roster of employees a fleet can dispatch to. Each Employee bundles a persona, a
registry builder (its ACI), and a default task prompt. The supervisor reads `description`
to decide who gets which subtask."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.skills import kb_digest, research
from anvil_ai_employee.tools import EmployeeContext


@dataclass
class Employee:
    name: str
    description: str
    persona: str
    build_registry: Callable[[EmployeeContext], ToolRegistry]
    task_prompt: str = "现在开始执行你被指派的任务。"


EMPLOYEES: dict[str, Employee] = {
    "kb_reporter": Employee(
        name="kb_reporter",
        description="知识库周报员:汇总知识库新增内容,产结构化中文周报。",
        persona=kb_digest.PERSONA,
        build_registry=kb_digest.build_registry,
        task_prompt="现在开始产出本期知识库周报。",
    ),
    "researcher": Employee(
        name="researcher",
        description="调研员:针对某主题在知识库做深入语义检索,产结构化调研纪要。",
        persona=research.PERSONA,
        build_registry=research.build_registry,
        task_prompt="现在开始针对你被指派的主题做调研。",
    ),
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_team.py -v`
Expected: 2 passed.

- [ ] **Step 7: Ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`
Expected: ruff clean; all pass.

- [ ] **Step 8: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/skills/research.py packages/ai-employee/src/anvil_ai_employee/fleet/ packages/ai-employee/tests/test_team.py
git -c user.email=robben766@users.noreply.github.com commit -m "feat(ai-employee): researcher skill + Employee fleet registry"
```

---

### Task 4: Generalize `worker.run_once` to pick employee by `job.employee`

**Files:**
- Modify: `packages/ai-employee/src/anvil_ai_employee/worker.py`
- Test: `packages/ai-employee/tests/test_worker.py` (extend)

The worker must: pick the `Employee` by `job.employee` (fallback `"kb_reporter"` → preserves M1), use `job.payload.get("task")` as the agent task (fallback the employee's `task_prompt`), and keep the existing "fail if the agent never completed the job" guard. M1's `kb_digest` skill path stays intact.

- [ ] **Step 1: Write the failing test**

Append to `packages/ai-employee/tests/test_worker.py` (this test mirrors the existing M1 worker test's respx setup — read the top of the file for the existing `_mock_gateway`/fixtures and reuse the SAME pattern; the assertion below is the new behavior):

```python
async def test_worker_runs_researcher_employee(session_factory, respx_mock):
    """A job tagged employee=researcher must run the researcher persona, not kb_reporter.
    We assert the agent was driven with the researcher's system prompt by inspecting the
    request body sent to the gateway."""
    import json
    import re

    import httpx

    from anvil_ai_employee.scheduler.queue import enqueue
    from anvil_ai_employee.worker import run_once

    captured = {}

    def _capture(request):
        body = json.loads(request.content)
        captured.setdefault("system", body["messages"][0]["content"])
        # First call: model emits submit_report to finish immediately.
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_report",
                                        "arguments": json.dumps(
                                            {"markdown": "调研纪要", "covered_until_iso": "2026-06-09T00:00:00"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    respx_mock.route(method="POST", url__regex=r".*chat/completions").mock(side_effect=_capture)

    await enqueue(
        session_factory,
        skill="kb_digest",
        payload={"task": "调研向量检索"},
        employee="researcher",
    )
    ran = await run_once(session_factory, model="deepseek-chat", worker_id="w1")
    assert ran is True
    assert "调研员" in captured["system"]  # researcher persona, not kb_reporter
```

> If the existing `test_worker.py` uses a different gateway-mock helper, REUSE that helper instead of the inline `respx_mock` above — keep the new test consistent with the file's established style. The behavioral assertion (`"调研员" in system`) is what matters.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_worker.py -k researcher -v`
Expected: FAIL — worker still uses the hardcoded `kb_reporter` persona, so `"调研员"` is not in the captured system prompt.

- [ ] **Step 3: Generalize `run_once`**

Edit `worker.py`. Replace the body that builds persona/registry/state with employee-aware selection. Specifically:

Replace the import + `SKILLS`/`TASK_PROMPT` region:

```python
from anvil_ai_employee.fleet.team import EMPLOYEES
from anvil_ai_employee.scheduler.queue import claim_one, fail
from anvil_ai_employee.tools import EmployeeContext
```

(Remove the `from anvil_ai_employee.skills import kb_digest` import and the `SKILLS = {...}` / `TASK_PROMPT = ...` module constants — they are replaced by `EMPLOYEES`.)

Then inside `run_once`, replace the skill-lookup + context/registry/state construction block with:

```python
        try:
            employee_name = job.employee or "kb_reporter"
            emp = EMPLOYEES.get(employee_name)
            if emp is None:
                await fail(session_factory, job.id, error=f"unknown employee: {employee_name}")
                return True
            ctx = EmployeeContext(
                session_factory=session_factory, employee=employee_name, job_id=job.id
            )
            registry = emp.build_registry(ctx)
            task = job.payload.get("task") or emp.task_prompt
            with tempfile.TemporaryDirectory() as workdir:
                state = AgentState.new(
                    system=emp.persona, task=task, workdir=workdir, max_steps=max_steps
                )
                tc = ToolContext(workdir=workdir)
                await agent_run(state, model, registry, tc)
            if await _job_status(session_factory, job.id) != "done":
                await fail(
                    session_factory,
                    job.id,
                    error="agent finished without calling submit_report",
                )
        except Exception as e:  # noqa: BLE001 — isolate one job's failure
            await fail(session_factory, job.id, error=f"{type(e).__name__}: {e}")
    return True
```

> Note: the `with span(...)` wrapper, the `claim_one` call, and the empty-queue early return stay exactly as in M1. Only the inner skill→employee selection changes. The `span` tag can stay `skill=job.skill`; optionally add `employee=employee_name`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_worker.py -v`
Expected: all pass — the new researcher test AND the existing M1 worker test (which enqueues without `employee` → falls back to `kb_reporter`).

- [ ] **Step 5: Ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`
Expected: ruff clean; all pass.

- [ ] **Step 6: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/worker.py packages/ai-employee/tests/test_worker.py
git -c user.email=robben766@users.noreply.github.com commit -m "feat(ai-employee): worker selects employee by job.employee (fleet-aware, M1 fallback preserved)"
```

---

### Task 5: `supervisor.py` — decompose + fan_out

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/fleet/supervisor.py`
- Test: `packages/ai-employee/tests/test_supervisor.py`

- [ ] **Step 1: Write the failing test**

```python
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
    subs = await decompose("调研并产周报", model="deepseek-chat", employees=["researcher", "kb_reporter"])
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
                "choices": [{"message": {"role": "assistant", "content": json.dumps({"subtasks": []})}}],
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

    from sqlalchemy import select

    from anvil_ai_employee.db import JobRow
    from anvil_ai_employee.fleet.supervisor import SubTask, fan_out

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_supervisor.py -v`
Expected: FAIL — `No module named 'anvil_ai_employee.fleet.supervisor'`.

- [ ] **Step 3: Implement `supervisor.py`**

```python
# packages/ai-employee/src/anvil_ai_employee/fleet/supervisor.py
"""Supervisor: decompose a goal into independent subtasks (one per employee) and fan them
out onto the M1 PG queue. Decomposition uses guard.structured_chat (json mode); illegal or
empty plans fall back so the fleet never spins on nothing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from anvil_guard import StructuredOutputError, structured_chat
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.scheduler.queue import enqueue

# Each employee maps to the M1 skill its worker registry is built from. Fleet jobs are
# tagged with both employee (selects persona/registry) and this skill (M1 compatibility).
_EMPLOYEE_SKILL = "kb_digest"


@dataclass
class SubTask:
    employee: str
    task: str


_DECOMPOSE_PROMPT = """你是一个团队主管。把下面的总目标拆解成若干条**相互独立**的子任务,每条指派给一名员工。

可用员工(name: 能力):
{roster}

只能指派给上面列出的员工 name。输出 JSON,格式:
{{"subtasks": [{{"employee": "<员工name>", "task": "<该员工要做的具体子任务>"}}, ...]}}

总目标:{goal}"""


async def decompose(goal: str, *, model: str, employees: list[str]) -> list[SubTask]:
    roster = "\n".join(f"- {e}" for e in employees)
    prompt = _DECOMPOSE_PROMPT.format(roster=roster, goal=goal)
    try:
        data = await structured_chat(
            model,
            [{"role": "user", "content": prompt}],
            schema={"required": ["subtasks"]},
        )
        raw = data.get("subtasks") or []
    except StructuredOutputError:
        raw = []
    allowed = set(employees)
    subs = [
        SubTask(employee=item["employee"], task=item["task"])
        for item in raw
        if isinstance(item, dict)
        and item.get("employee") in allowed
        and item.get("task")
    ]
    if not subs:
        # Never fan out nothing: assign the whole goal to the first employee.
        subs = [SubTask(employee=employees[0], task=goal)]
    return subs


async def fan_out(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    goal_id: uuid.UUID,
    subtasks: list[SubTask],
) -> list[uuid.UUID]:
    ids: list[uuid.UUID] = []
    for sub in subtasks:
        jid = await enqueue(
            session_factory,
            skill=_EMPLOYEE_SKILL,
            payload={"task": sub.task},
            goal_id=goal_id,
            employee=sub.employee,
        )
        ids.append(jid)
    return ids
```

> NOTE (signature confirmed): `structured_chat(model, messages, *, schema={"required": [...]}) -> dict` (see `packages/core/guard/src/anvil_guard/structured.py:32`; `StructuredOutputError` and `structured_chat` are both top-level exports of `anvil_guard`, as imported by `memory/mem0.py:22`). The code above matches. Because `response_format=json_object` is used, the prompt MUST contain the literal word "json" — `_DECOMPOSE_PROMPT` already says "输出 JSON". The test mocks the gateway HTTP response directly, so it is signature-agnostic regardless.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_supervisor.py -v`
Expected: 4 passed.

- [ ] **Step 5: Ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`
Expected: ruff clean; all pass.

- [ ] **Step 6: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/fleet/supervisor.py packages/ai-employee/tests/test_supervisor.py
git -c user.email=robben766@users.noreply.github.com commit -m "feat(ai-employee): supervisor decompose (structured_chat) + fan_out"
```

---

### Task 6: `aggregator.py` — children_terminal + aggregate

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/fleet/aggregator.py`
- Test: `packages/ai-employee/tests/test_aggregator.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/ai-employee/tests/test_aggregator.py
import json
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
    from sqlalchemy import select

    from anvil_ai_employee.db import GoalRow
    from anvil_ai_employee.fleet.aggregator import aggregate

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_aggregator.py -v`
Expected: FAIL — `No module named 'anvil_ai_employee.fleet.aggregator'`.

- [ ] **Step 3: Implement `aggregator.py`**

```python
# packages/ai-employee/src/anvil_ai_employee/fleet/aggregator.py
"""Aggregator: once every child job of a goal is terminal (done or failed), synthesize
their outputs into the goal's final result. Failed subtasks are included and labelled so
the final deliverable honestly reflects what did not complete (ACI, at the fleet level)."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_obs import span

from anvil_gateway import chat
from anvil_ai_employee.db import GoalRow, JobRow

_TERMINAL = {"done", "failed"}


async def children_terminal(
    session_factory: async_sessionmaker[AsyncSession], goal_id: uuid.UUID
) -> bool:
    async with session_factory() as s:
        rows = (
            (await s.execute(select(JobRow).where(JobRow.goal_id == goal_id))).scalars().all()
        )
    if not rows:
        return True  # no children → trivially terminal
    return all(r.status in _TERMINAL for r in rows)


async def aggregate(
    session_factory: async_sessionmaker[AsyncSession],
    goal_id: uuid.UUID,
    *,
    model: str,
) -> str | None:
    """Return the synthesized final result (and persist it) if all children are terminal;
    otherwise return None without writing. Idempotent: safe to call repeatedly."""
    if not await children_terminal(session_factory, goal_id):
        return None
    async with session_factory() as s:
        goal = (await s.execute(select(GoalRow).where(GoalRow.id == goal_id))).scalar_one()
        children = (
            (await s.execute(select(JobRow).where(JobRow.goal_id == goal_id))).scalars().all()
        )
    parts = []
    for c in children:
        if c.status == "done":
            parts.append(f"### 员工 {c.employee} 的产出\n{c.result or ''}")
        else:
            parts.append(f"### 员工 {c.employee}(未完成: {c.error or '失败'})")
    synthesis_prompt = (
        f"总目标:{goal.objective}\n\n以下是各员工的子任务产出,请综合成一份连贯的最终中文交付物;"
        f"若某员工未完成,在最终产出中如实标注该部分缺失。\n\n" + "\n\n".join(parts)
    )
    with span("ai_employee.fleet.aggregate", goal=str(goal_id)):
        resp = await chat(model, [{"role": "user", "content": synthesis_prompt}])
        final = resp.raw["choices"][0]["message"]["content"] or ""
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                update(GoalRow)
                .where(GoalRow.id == goal_id)
                .values(status="done", result=final)
            )
    return final
```

> NOTE (confirmed): `chat(model, messages)` returns an object whose `.raw["choices"][0]["message"]["content"]` holds the text (same access as `hitl.py:82`); `tools=` is optional and omitted here. `JobRow.error` already exists (M1). `GoalRow` has no `finished_at` write above — that's fine (nullable); optionally also set `finished_at=func.now()` in the update for completeness.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_aggregator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`
Expected: ruff clean; all pass.

- [ ] **Step 6: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/fleet/aggregator.py packages/ai-employee/tests/test_aggregator.py
git -c user.email=robben766@users.noreply.github.com commit -m "feat(ai-employee): aggregator — synthesize goal result when all children terminal"
```

---

### Task 7: CLI `team run` / `team status`

**Files:**
- Modify: `packages/ai-employee/src/anvil_ai_employee/cli.py`
- Test: `packages/ai-employee/tests/test_ai_employee_cli.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `packages/ai-employee/tests/test_ai_employee_cli.py`:

```python
@pytest.mark.asyncio
async def test_team_run_creates_goal_and_children(session_factory, respx_mock):
    import json
    import uuid

    import httpx
    from sqlalchemy import select

    from anvil_ai_employee.cli import team_run
    from anvil_ai_employee.db import GoalRow, JobRow

    plan = {"subtasks": [{"employee": "researcher", "task": "调研 X"}, {"employee": "kb_reporter", "task": "写周报"}]}
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


@pytest.mark.asyncio
async def test_team_status_text_lists_children(session_factory):
    import uuid

    from anvil_ai_employee.cli import team_status_text
    from anvil_ai_employee.db import GoalRow, JobRow

    gid = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            s.add(GoalRow(id=gid, objective="g", status="running"))
            s.add(JobRow(skill="kb_digest", payload={"task": "a"}, status="done", result="r", goal_id=gid, employee="researcher"))
            s.add(JobRow(skill="kb_digest", payload={"task": "b"}, status="pending", goal_id=gid, employee="kb_reporter"))
    text = await team_status_text(session_factory, goal_id=gid, model="deepseek-chat")
    assert "researcher" in text and "kb_reporter" in text
    assert "done" in text and "pending" in text
    # not all terminal → no synthesis yet
    assert "最终产出" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_ai_employee_cli.py -k "team_run or team_status" -v`
Expected: FAIL — `cannot import name 'team_run'`.

- [ ] **Step 3: Add CLI helpers to `cli.py`**

Add near the other async helpers (before `def main()`):

```python
async def team_run(
    sf: async_sessionmaker[AsyncSession], *, objective: str, model: str
) -> uuid.UUID:
    from anvil_ai_employee.db import GoalRow
    from anvil_ai_employee.fleet.supervisor import decompose, fan_out
    from anvil_ai_employee.fleet.team import EMPLOYEES

    goal_id = uuid.uuid4()
    async with sf() as s:
        async with s.begin():
            s.add(GoalRow(id=goal_id, objective=objective, status="running"))
    subs = await decompose(objective, model=model, employees=list(EMPLOYEES))
    await fan_out(sf, goal_id=goal_id, subtasks=subs)
    return goal_id


async def team_status_text(
    sf: async_sessionmaker[AsyncSession], *, goal_id: uuid.UUID, model: str
) -> str:
    from sqlalchemy import select

    from anvil_ai_employee.db import GoalRow, JobRow
    from anvil_ai_employee.fleet.aggregator import aggregate

    async with sf() as s:
        goal = (await s.execute(select(GoalRow).where(GoalRow.id == goal_id))).scalar_one_or_none()
        if goal is None:
            return f"goal {goal_id} 不存在。"
        children = (
            (await s.execute(select(JobRow).where(JobRow.goal_id == goal_id))).scalars().all()
        )
    lines = [f"goal {str(goal_id)[:8]}: {goal.objective}  status={goal.status}"]
    for c in children:
        lines.append(f"  [{c.employee}] {c.status}  task={c.payload.get('task', '')}")
    final = await aggregate(sf, goal_id, model=model)
    if final is not None:
        lines.append("\n=== 最终产出 ===\n" + final)
    return "\n".join(lines)
```

- [ ] **Step 4: Register the subcommands in `main()`**

After the `run-mcp` parser block, add:

```python
    team_p = sub.add_parser("team")
    team_sub = team_p.add_subparsers(dest="team_cmd", required=True)

    tr = team_sub.add_parser("run")
    tr.add_argument("--goal", required=True)
    tr.add_argument("--model", default="deepseek-chat")

    ts = team_sub.add_parser("status")
    ts.add_argument("--goal", required=True)
    ts.add_argument("--model", default="deepseek-chat")
```

And in the dispatch section, add:

```python
    elif args.cmd == "team":
        if args.team_cmd == "run":
            gid = asyncio.run(team_run(sf, objective=args.goal, model=args.model))
            print(f"goal_id = {gid}")
            print(asyncio.run(team_status_text(sf, goal_id=gid, model=args.model)))
        elif args.team_cmd == "status":
            print(
                asyncio.run(
                    team_status_text(sf, goal_id=uuid.UUID(args.goal), model=args.model)
                )
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd packages/ai-employee && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest tests/test_ai_employee_cli.py -k "team" -v`
Expected: 2 passed.

- [ ] **Step 6: Ruff + whole-package suite**

Run: `cd packages/ai-employee && uv run ruff check --fix src tests && ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil uv run pytest -m "not live" -q`
Expected: ruff clean; all pass.

- [ ] **Step 7: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/cli.py packages/ai-employee/tests/test_ai_employee_cli.py
git -c user.email=robben766@users.noreply.github.com commit -m "feat(ai-employee): CLI team run/status (fan out a goal, aggregate when done)"
```

---

### Task 8: Example 12 + CLAUDE.md M5 section

**Files:**
- Create: `examples/12-ai-employee-fleet/README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the example README**

```markdown
# 12 — AI 员工·多员工编队(P4-M5:集大成收尾)

M1-M4 把**单个员工**的全链路打通(PG 队列定时触发 + 三层记忆 + HITL 防跑飞 + MCP 外部工具)。M5 把"一个员工"扩成"一队员工":**supervisor 拆目标 → 扇出给不同员工 → 一队 worker 并行干 → aggregator 综合**。

## 不新建编排引擎,直接复用 M1 的 PG 队列

"编队"的工作分发 = M1 那张 `ae_jobs` 队列 + `SELECT … FOR UPDATE SKIP LOCKED`。M1 已证明多 worker 并行抢 job 不会重复领取——所以 fleet 层不需要新调度器,只要:给 job 标上 `goal_id`(属于哪个目标)和 `employee`(指派给谁),worker 按 `employee` 选对应的 persona/工具集即可。多开几个 `work --loop` 进程就是一队并行的员工。

## supervisor:拆解 + 扇出

`team run --goal "<目标>"` → supervisor 用 `guard.structured_chat`(json 模式)把目标拆成相互独立的子任务,每条指派给一名员工 → 逐个 `enqueue`(带 goal_id + employee)。拆解非法/空 → 兜底成单任务交给第一个员工,绝不空转。

## aggregator:子任务全终态后综合

`team status --goal <id>` → 列各子任务状态;一旦所有子 job 都 done/failed,把各员工产出喂给 LLM 综合成最终交付物,落 `ae_goals.result`。**失败的子任务也纳入综合并标注缺失**——最终产出诚实反映哪部分没成(ACI 延伸到编队层)。aggregate 幂等,没全完成就返回 None 不写。

## 跑一遍(需 `ANVIL_DATABASE_URL` + `DEEPSEEK_API_KEY`)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil

# 1) 下目标 → supervisor 拆解 + 扇出子任务给不同员工
uv run anvil-ai-employee team run --goal "调研'向量检索'并产一份知识库周报"
# → goal_id = ...
# → goal ...: ... status=running
#     [researcher] pending  task=调研向量检索
#     [kb_reporter] pending  task=产周报

# 2) 起一队 worker 并行处理(多开几个就是并行编队)
uv run anvil-ai-employee work --loop &
uv run anvil-ai-employee work --loop &

# 3) 看进度;子任务全完成时自动综合出最终交付物
uv run anvil-ai-employee team status --goal <goal_id>
# → [researcher] done ...  [kb_reporter] done ...
#   === 最终产出 ===
#   <综合后的交付物>
```

## 一队不同的员工

`EMPLOYEES` 注册表里 `kb_reporter`(周报员)和 `researcher`(调研员)角色不同(persona/职责不同),supervisor 按各自能力简介分派子任务。worker 按 `job.employee` 选谁上场——M2 记忆、M3 HITL、M4 MCP 工具的员工都能直接作为编队成员加入(各自带能力)。

## 复用与新建

- ✓ 真复用:M1 `scheduler/queue`(enqueue/claim SKIP LOCKED 语义不变)+ `worker` 骨架 + P3 harness;`guard.structured_chat`(拆解);`gateway.chat`(综合);M2/M3/M4 员工不改。
- ✗ 新建:`ae_goals` 表 + `JobRow.goal_id/employee`;`fleet/`(supervisor/aggregator/team);worker 的 employee 泛化;CLI `team` 子命令。

## 留待(螺旋)
子任务 DAG 依赖、员工间消息/协商/移交、动态扩缩容、陪审团(P2)择优综合、goal 级 HITL。至此 P4(AI 员工)与四产品主体全部完成。
```

- [ ] **Step 2: Add the M5 subsection to `CLAUDE.md`**

Find the `anvil-ai-employee` section and add an M5 subsection after the M4 one, matching the sibling heading depth (the M2a/M2b/M3/M4 subsections use `###`). Content:

```markdown
### M5「多员工编队」(examples/12)

- **不新建编排引擎,直接复用 M1 PG 队列**:fleet 工作分发 = `ae_jobs` + SKIP LOCKED(M1 已证多 worker 无重复领取);job 加 `goal_id`(属哪个目标)+ `employee`(指派给谁),worker 按 `employee` 选 persona/registry。多开 `work --loop` = 并行编队。
- **supervisor**(`fleet/supervisor.py`):`decompose` 用 `guard.structured_chat` 把目标拆成独立子任务(非法/空→兜底单任务,绝不空转);`fan_out` 逐个 `enqueue`(带 goal_id+employee)。
- **aggregator**(`fleet/aggregator.py`):`children_terminal` 判全 done/failed;`aggregate` 综合各产出写 `ae_goals.result`,**失败子任务也纳入并标注缺失**(ACI 延伸到编队层),未全终态返 None、幂等。
- **team**(`fleet/team.py`):`EMPLOYEES` 注册表(kb_reporter 周报员 + researcher 调研员,角色异构);worker 泛化按 `job.employee` 选员工,`job.payload['task']` 作子任务(fallback 保 M1 行为)。
- 存储:`ae_goals` 表 + `JobRow.goal_id/employee`(nullable,M1-M4 单 job 路径不破)。CLI `team run --goal`/`team status --goal`。M2/M3/M4 员工天然可作编队成员。
- 边界:子任务 DAG 依赖、员工间消息/协商、动态扩缩容、陪审团择优综合留作螺旋。**至此 P4 与四产品主体全部完成。**
```

- [ ] **Step 3: Commit**

```bash
git add examples/12-ai-employee-fleet/README.md CLAUDE.md
git -c user.email=robben766@users.noreply.github.com commit -m "docs(ai-employee): M5 multi-employee fleet example + CLAUDE.md section"
```

---

## Final Verification (after all tasks)

- [ ] **Whole-repo suite green (CI parity):** from repo root `uv run pytest -m "not live" -q`. Expected: all pass, no collection errors.
- [ ] **Optional live end-to-end:** with `DEEPSEEK_API_KEY` + PG, run the example README flow (`team run` → `work` → `team status`) and confirm decompose → parallel run → synthesized result, including a goal where one subtask fails (honest aggregation labels the gap).
- [ ] **Dispatch final code reviewer (opus)** for the whole M5 branch — focus: M1 backward-compat (queue/worker still pass), decompose fallback safety, aggregate idempotency, no leak of fleet columns into M1-M4 single-job paths.
- [ ] **Use superpowers:finishing-a-development-branch** to open the PR, wait for CI green, self-merge.

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| `ae_goals` table + `JobRow.goal_id/employee` | Task 1 |
| `enqueue`/`claim_one`/`ClaimedJob` carry goal_id + employee | Task 2 |
| `Employee` registry (≥2 distinct employees) | Task 3 |
| `researcher` skill | Task 3 |
| worker selects employee by `job.employee` (M1 fallback) | Task 4 |
| `supervisor.decompose` (structured_chat, filter illegal, empty fallback) | Task 5 |
| `supervisor.fan_out` | Task 5 |
| `aggregator.children_terminal` | Task 6 |
| `aggregator.aggregate` (synthesize, label failures, idempotent) | Task 6 |
| CLI `team run` / `team status` | Task 7 |
| Example 12 + CLAUDE.md | Task 8 |
| Reuse M1 queue/worker + P3 harness; M2/M3/M4 unchanged | Tasks 2, 4 (additive) |
