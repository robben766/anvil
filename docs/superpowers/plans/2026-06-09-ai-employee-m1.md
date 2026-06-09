# P4-M1 AI 员工「知识库周报员」实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cron 定时唤醒"知识库周报员",在进程内跑 P3 agent 循环,用 P1 知识库工具读"上次报告后新入库"的内容,产出结构化摘要并把覆盖时间点写回最小长期记忆。

**Architecture:** 新包 `packages/ai-employee`(`anvil_ai_employee`)。PG 原生 `SKIP LOCKED` 队列 + croniter 触发器 + 最小长期记忆(报告标记)+ worker 复用 `anvil_code_agent.harness.run` 跑循环。周报员工具(kb_recent/kb_search/recall_marker/submit_report)用 P3 `@tool` 协议,但因 P3 工具是同步而 DB/检索是异步,工具内用 `ThreadPoolExecutor + asyncio.run` 桥接(复用 M6 模式)。

**Tech Stack:** Python 3.12 / SQLAlchemy 异步 + pgvector / croniter / 复用 anvil-gateway、anvil-obs、anvil-code-agent、anvil-kb。PG@5434,无 SQLite、无 Redis。

---

## 关键既有接口(实现时照此调用,勿臆造)

- `anvil_code_agent.state.AgentState.new(*, system, task, workdir, max_steps)` → 不可变状态。
- `anvil_code_agent.harness.loop.run(state, model, registry, ctx, *, policy=auto_approve, token_budget=None, summarizer=None)` → 异步,驱动到 done/exhausted。
- `anvil_code_agent.tools.base`:`Tool`、`ToolResult(content, ok, truncated=False)`、`ToolContext(workdir, timeout=120.0, max_output=4096, executor=None)`、`ToolRegistry(list[Tool])`、`tool(*, name, description, params, required)` 装饰器。**工具 fn 签名:`fn(args: dict, ctx: ToolContext) -> ToolResult`,同步。** registry.dispatch 已捕获异常转 ok=False。
- `anvil_kb.db.make_session_factory(database_url=None)` → `async_sessionmaker[AsyncSession]`(读 `ANVIL_DATABASE_URL`);`anvil_kb.db.make_engine`;`anvil_kb.db.DocumentRow`(字段:id, title, source_name, content, created_at)。
- `anvil_kb.embed.FastEmbedEmbedder()`;`anvil_kb.store.pg.PgVectorStore(session_factory)`;`anvil_kb.retrieve.retriever.Retriever(embedder, store, mode="dense")`;`Retriever.retrieve(question, k) -> list[ScoredChunk]`(ScoredChunk 有 `.chunk.content`、`.score`)。
- `anvil_obs.span(name, **attrs)` 上下文管理器。

---

## File Structure

- `packages/ai-employee/pyproject.toml` — 包定义 + 依赖 + script
- `packages/ai-employee/src/anvil_ai_employee/__init__.py`
- `packages/ai-employee/src/anvil_ai_employee/db.py` — Base + 三张 Row + session 工厂复用
- `packages/ai-employee/src/anvil_ai_employee/migrations/` — Alembic(或单文件 DDL helper,见 Task 2)
- `packages/ai-employee/src/anvil_ai_employee/scheduler/queue.py` — PG 队列
- `packages/ai-employee/src/anvil_ai_employee/scheduler/trigger.py` — Trigger 协议 + CronTrigger
- `packages/ai-employee/src/anvil_ai_employee/memory/store.py` — MemoryStore
- `packages/ai-employee/src/anvil_ai_employee/asyncbridge.py` — `block_on(coro)` 同步桥
- `packages/ai-employee/src/anvil_ai_employee/tools.py` — EmployeeContext + 四个 @tool
- `packages/ai-employee/src/anvil_ai_employee/skills/kb_digest.py` — persona + build_registry
- `packages/ai-employee/src/anvil_ai_employee/worker.py` — run_once
- `packages/ai-employee/src/anvil_ai_employee/cli.py` — CLI
- `packages/ai-employee/tests/...` — 每模块对应测试
- 根 `pyproject.toml` `members` 增 `packages/ai-employee`
- `examples/07-ai-employee/README.md`

---

## Task 1: 包脚手架 + db.py 三张表

**Files:**
- Create: `packages/ai-employee/pyproject.toml`
- Create: `packages/ai-employee/src/anvil_ai_employee/__init__.py`
- Create: `packages/ai-employee/src/anvil_ai_employee/db.py`
- Modify: 根 `pyproject.toml`(`members` 增 `"packages/ai-employee"`)
- Test: `packages/ai-employee/tests/test_db.py`

- [ ] **Step 1: 写 pyproject.toml**

```toml
[project]
name = "anvil-ai-employee"
version = "0.1.0"
description = "anvil: AI employee — cron-triggered agents with PG queue + long-term memory (P4)"
requires-python = ">=3.12"
dependencies = [
    "anvil-gateway",
    "anvil-obs",
    "anvil-code-agent",
    "anvil-kb",
    "sqlalchemy>=2",
    "asyncpg>=0.29",
    "pgvector>=0.3",
    "croniter>=2",
]

[project.scripts]
anvil-ai-employee = "anvil_ai_employee.cli:main"

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "respx>=0.21", "httpx>=0.27", "ruff>=0.6"]

[tool.uv.sources]
anvil-gateway = { workspace = true }
anvil-obs = { workspace = true }
anvil-code-agent = { workspace = true }
anvil-kb = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/anvil_ai_employee"]
```

`__init__.py` 空文件(含模块 docstring 一行)。

- [ ] **Step 2: 根 pyproject members 增本包**

把根 `pyproject.toml` 的 `members = [...]` 改为包含 `"packages/ai-employee"`。改完跑 `uv sync --all-packages` 确认装得上。

- [ ] **Step 3: 写失败测试 test_db.py**

```python
import uuid
from datetime import UTC, datetime

import pytest

from anvil_ai_employee.db import JobRow, MemoryRow, ScheduleRow


def test_rows_have_expected_columns():
    # schedule
    cols = ScheduleRow.__table__.columns.keys()
    assert {"id", "name", "cron_expr", "skill", "payload", "next_run_at", "enabled", "created_at"} <= set(cols)
    # job
    jcols = JobRow.__table__.columns.keys()
    assert {"id", "schedule_id", "skill", "payload", "status", "result", "error", "locked_by",
            "created_at", "started_at", "finished_at"} <= set(jcols)
    # memory
    mcols = MemoryRow.__table__.columns.keys()
    assert {"id", "employee", "kind", "content", "created_at"} <= set(mcols)


def test_tablenames():
    assert ScheduleRow.__tablename__ == "ae_schedules"
    assert JobRow.__tablename__ == "ae_jobs"
    assert MemoryRow.__tablename__ == "ae_memories"
```

- [ ] **Step 4: 跑测试确认失败**

Run: `uv run pytest packages/ai-employee/tests/test_db.py -q`
Expected: FAIL(ImportError: anvil_ai_employee.db)

- [ ] **Step 5: 写 db.py**

```python
"""AI employee persistence: schedules, job queue, long-term memory (M1: report markers only)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Reuse kb's engine/session helpers — one ANVIL_DATABASE_URL, one PG.
from anvil_kb.db import make_engine, make_session_factory  # noqa: F401  (re-exported)


class Base(DeclarativeBase):
    pass


class ScheduleRow(Base):
    __tablename__ = "ae_schedules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    cron_expr: Mapped[str] = mapped_column(Text, nullable=False)
    skill: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    next_run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class JobRow(Base):
    __tablename__ = "ae_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    schedule_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    skill: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    locked_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MemoryRow(Base):
    __tablename__ = "ae_memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest packages/ai-employee/tests/test_db.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add packages/ai-employee/pyproject.toml packages/ai-employee/src pyproject.toml packages/ai-employee/tests/test_db.py
git commit -m "feat(ai-employee): package scaffold + schedules/jobs/memories tables"
```

---

## Task 2: 建表(测试夹具 create_all)

> M1 不引入 Alembic 迁移链(anvil 各包测试直接 `Base.metadata.create_all`);建表用一个 fixture helper,与 kb 测试风格一致。后续真要 Alembic 在 P4 后段补。

**Files:**
- Create: `packages/ai-employee/tests/conftest.py`
- Test: `packages/ai-employee/tests/test_schema_create.py`

- [ ] **Step 1: 写 conftest.py(建表 + 清表 fixture)**

```python
import os

import pytest
import pytest_asyncio

from anvil_ai_employee.db import Base, make_engine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def engine():
    url = os.environ.get("ANVIL_DATABASE_URL")
    if not url:
        pytest.skip("ANVIL_DATABASE_URL not set (needs real PG@5434)")
    eng = make_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

- [ ] **Step 2: 写 test_schema_create.py(冒烟:三表可建可查)**

```python
import pytest
from sqlalchemy import select

from anvil_ai_employee.db import JobRow

pytestmark = pytest.mark.asyncio


async def test_tables_created_and_queryable(session_factory):
    async with session_factory() as s:
        rows = (await s.execute(select(JobRow))).scalars().all()
        assert rows == []
```

- [ ] **Step 3: 跑测试**

Run: `ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test uv run pytest packages/ai-employee/tests/test_schema_create.py -q`
Expected: PASS(无 PG 则 skip)

- [ ] **Step 4: Commit**

```bash
git add packages/ai-employee/tests/conftest.py packages/ai-employee/tests/test_schema_create.py
git commit -m "test(ai-employee): schema create/drop fixtures on real PG"
```

---

## Task 3: PG 队列(SKIP LOCKED)

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/scheduler/__init__.py`(空)
- Create: `packages/ai-employee/src/anvil_ai_employee/scheduler/queue.py`
- Test: `packages/ai-employee/tests/test_queue.py`

- [ ] **Step 1: 写失败测试 test_queue.py**

```python
import asyncio

import pytest
from sqlalchemy import select

from anvil_ai_employee.db import JobRow
from anvil_ai_employee.scheduler.queue import claim_one, complete, enqueue, fail

pytestmark = pytest.mark.asyncio


async def test_enqueue_then_claim_marks_running(session_factory):
    job_id = await enqueue(session_factory, skill="kb_digest", payload={"a": 1})
    claimed = await claim_one(session_factory, worker_id="w1")
    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.status == "running"
    assert claimed.locked_by == "w1"
    assert claimed.started_at is not None


async def test_claim_empty_returns_none(session_factory):
    assert await claim_one(session_factory, worker_id="w1") is None


async def test_concurrent_claims_no_double_take(session_factory):
    ids = {await enqueue(session_factory, skill="kb_digest", payload={}) for _ in range(5)}
    # 8 workers race for 5 jobs
    results = await asyncio.gather(
        *[claim_one(session_factory, worker_id=f"w{i}") for i in range(8)]
    )
    claimed = [r for r in results if r is not None]
    assert len(claimed) == 5
    assert {c.id for c in claimed} == ids  # each job taken exactly once


async def test_complete_and_fail(session_factory):
    j1 = await enqueue(session_factory, skill="s", payload={})
    c1 = await claim_one(session_factory, worker_id="w1")
    await complete(session_factory, c1.id, result="done!")
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == j1))).scalar_one()
        assert row.status == "done" and row.result == "done!" and row.finished_at is not None

    j2 = await enqueue(session_factory, skill="s", payload={})
    c2 = await claim_one(session_factory, worker_id="w1")
    await fail(session_factory, c2.id, error="boom")
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == j2))).scalar_one()
        assert row.status == "failed" and row.error == "boom" and row.finished_at is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_queue.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 queue.py**

```python
"""PG-native job queue. Workers claim with FOR UPDATE SKIP LOCKED — no Redis, one PG."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from anvil_ai_employee.db import JobRow


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    skill: str
    payload: dict[str, Any]
    status: str
    locked_by: str
    started_at: Any


async def enqueue(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    skill: str,
    payload: dict[str, Any],
    schedule_id: uuid.UUID | None = None,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                insert(JobRow).values(
                    id=job_id, schedule_id=schedule_id, skill=skill,
                    payload=payload, status="pending",
                )
            )
    return job_id


async def claim_one(
    session_factory: async_sessionmaker[AsyncSession], *, worker_id: str
) -> ClaimedJob | None:
    """Atomically grab one pending job. Concurrent workers never take the same row."""
    async with session_factory() as s:
        async with s.begin():
            row = (
                await s.execute(
                    select(JobRow)
                    .where(JobRow.status == "pending")
                    .order_by(JobRow.created_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            await s.execute(
                update(JobRow)
                .where(JobRow.id == row.id)
                .values(status="running", locked_by=worker_id, started_at=func.now())
            )
            # re-read for the populated started_at
            fresh = (
                await s.execute(select(JobRow).where(JobRow.id == row.id))
            ).scalar_one()
            return ClaimedJob(
                id=fresh.id, skill=fresh.skill, payload=fresh.payload,
                status=fresh.status, locked_by=fresh.locked_by, started_at=fresh.started_at,
            )


async def complete(
    session_factory: async_sessionmaker[AsyncSession], job_id: uuid.UUID, *, result: str
) -> None:
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                update(JobRow).where(JobRow.id == job_id).values(
                    status="done", result=result, finished_at=func.now()
                )
            )


async def fail(
    session_factory: async_sessionmaker[AsyncSession], job_id: uuid.UUID, *, error: str
) -> None:
    async with session_factory() as s:
        async with s.begin():
            await s.execute(
                update(JobRow).where(JobRow.id == job_id).values(
                    status="failed", error=error, finished_at=func.now()
                )
            )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_queue.py -q`
Expected: PASS(并发用例证明 SKIP LOCKED 无重复领取)

- [ ] **Step 5: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/scheduler packages/ai-employee/tests/test_queue.py
git commit -m "feat(ai-employee): PG SKIP LOCKED job queue (enqueue/claim/complete/fail)"
```

---

## Task 4: CronTrigger

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/scheduler/trigger.py`
- Test: `packages/ai-employee/tests/test_trigger.py`

设计:`CronTrigger.due(session_factory, now)` 读所有 enabled 且 `next_run_at <= now` 的 schedule,为每个 enqueue 一个 job,并把 `next_run_at` 推进到 `croniter(cron_expr, now).get_next()`。enqueue 与推进在**同一事务**(避免重复入队)。`now` 由参数注入(不调 `datetime.now`,保证可测)。

- [ ] **Step 1: 写失败测试 test_trigger.py**

```python
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from anvil_ai_employee.db import JobRow, ScheduleRow
from anvil_ai_employee.scheduler.trigger import CronTrigger

pytestmark = pytest.mark.asyncio


async def _add_schedule(session_factory, *, name, cron, next_run_at, enabled=True):
    async with session_factory() as s:
        async with s.begin():
            s.add(ScheduleRow(
                name=name, cron_expr=cron, skill="kb_digest",
                payload={}, next_run_at=next_run_at, enabled=enabled,
            ))


async def test_due_enqueues_and_advances(session_factory):
    now = datetime(2026, 6, 9, 9, 0, tzinfo=UTC)
    # daily at 09:00; due exactly now
    await _add_schedule(session_factory, name="daily", cron="0 9 * * *",
                        next_run_at=now)
    trig = CronTrigger(session_factory)
    n = await trig.due(now)
    assert n == 1
    async with session_factory() as s:
        jobs = (await s.execute(select(JobRow))).scalars().all()
        assert len(jobs) == 1 and jobs[0].skill == "kb_digest"
        sched = (await s.execute(select(ScheduleRow))).scalar_one()
        # advanced to next 09:00 (tomorrow)
        assert sched.next_run_at == now + timedelta(days=1)


async def test_not_due_does_nothing(session_factory):
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    await _add_schedule(session_factory, name="daily", cron="0 9 * * *",
                        next_run_at=datetime(2026, 6, 9, 9, 0, tzinfo=UTC))
    trig = CronTrigger(session_factory)
    assert await trig.due(now) == 0
    async with session_factory() as s:
        assert (await s.execute(select(JobRow))).scalars().all() == []


async def test_disabled_skipped(session_factory):
    now = datetime(2026, 6, 9, 9, 0, tzinfo=UTC)
    await _add_schedule(session_factory, name="off", cron="0 9 * * *",
                        next_run_at=now, enabled=False)
    trig = CronTrigger(session_factory)
    assert await trig.due(now) == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_trigger.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 trigger.py**

```python
"""Triggers turn time/events into queued jobs. M1: CronTrigger. The Trigger protocol
keeps webhook/on-demand triggers pluggable without touching the worker."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from croniter import croniter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import ScheduleRow
from anvil_ai_employee.scheduler.queue import enqueue


class Trigger(Protocol):
    async def due(self, now: datetime) -> int:
        """Enqueue jobs that are due as of *now*; return how many were enqueued."""
        ...


class CronTrigger:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def due(self, now: datetime) -> int:
        enqueued = 0
        async with self._sf() as s:
            async with s.begin():
                rows = (
                    await s.execute(
                        select(ScheduleRow)
                        .where(ScheduleRow.enabled.is_(True))
                        .where(ScheduleRow.next_run_at <= now)
                        .with_for_update(skip_locked=True)
                    )
                ).scalars().all()
                for sched in rows:
                    await enqueue(
                        self._sf, skill=sched.skill, payload=sched.payload,
                        schedule_id=sched.id,
                    )
                    sched.next_run_at = croniter(sched.cron_expr, now).get_next(datetime)
                    enqueued += 1
        return enqueued
```

> 注意:`enqueue` 内部自开 session,在 `due` 的事务内调用是另一连接的独立事务——M1 单 ticker 假设下可接受(不会重复:next_run_at 在同事务推进,下一拍不会再选中)。若审查认为要严格同事务,可改为 due 内直接 `s.add(JobRow(...))`;实现者按测试通过为准,二选一并在 commit message 注明。

croniter 的 `get_next(datetime)` 返回带 tz 的 datetime(传入 now 带 tz 时)。

- [ ] **Step 4: 跑测试确认通过**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_trigger.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/scheduler/trigger.py packages/ai-employee/tests/test_trigger.py
git commit -m "feat(ai-employee): CronTrigger — croniter due() enqueues + advances next_run_at"
```

---

## Task 5: MemoryStore(最小长期记忆)

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/memory/__init__.py`(空)
- Create: `packages/ai-employee/src/anvil_ai_employee/memory/store.py`
- Test: `packages/ai-employee/tests/test_memory.py`

- [ ] **Step 1: 写失败测试 test_memory.py**

```python
import pytest

from anvil_ai_employee.memory.store import MemoryStore

pytestmark = pytest.mark.asyncio


async def test_write_then_last(session_factory):
    store = MemoryStore(session_factory)
    await store.write(employee="kb_reporter", kind="report_marker", content='{"covered_until": "2026-06-01T00:00:00+00:00"}')
    await store.write(employee="kb_reporter", kind="report_marker", content='{"covered_until": "2026-06-08T00:00:00+00:00"}')
    last = await store.last(employee="kb_reporter", kind="report_marker")
    assert last is not None and "2026-06-08" in last


async def test_last_none_when_empty(session_factory):
    store = MemoryStore(session_factory)
    assert await store.last(employee="kb_reporter", kind="report_marker") is None


async def test_employee_isolation(session_factory):
    store = MemoryStore(session_factory)
    await store.write(employee="a", kind="report_marker", content="A")
    await store.write(employee="b", kind="report_marker", content="B")
    assert await store.last(employee="a", kind="report_marker") == "A"
    assert await store.last(employee="b", kind="report_marker") == "B"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_memory.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 store.py**

```python
"""Minimal long-term memory. M1: only 'report_marker' — remembers where the reporter
left off so it never repeats. M2 grows this into the full three-tier memory."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import MemoryRow


class MemoryStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def write(self, *, employee: str, kind: str, content: str) -> None:
        async with self._sf() as s:
            async with s.begin():
                s.add(MemoryRow(employee=employee, kind=kind, content=content))

    async def last(self, *, employee: str, kind: str) -> str | None:
        """Most recent memory of *kind* for *employee* (pure recency; no vectors in M1)."""
        async with self._sf() as s:
            row = (
                await s.execute(
                    select(MemoryRow)
                    .where(MemoryRow.employee == employee)
                    .where(MemoryRow.kind == kind)
                    .order_by(MemoryRow.created_at.desc(), MemoryRow.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            return row.content if row else None
```

> 注意:同一测试里两次 write 的 `created_at` 可能落在同一毫秒,故 `order_by` 加 `id.desc()` 兜底——但 uuid 无序,不能保证插入顺序。**实现者改进:给 `last` 的排序加一个单调来源**——最稳妥是 MemoryRow 增一个自增 `seq`(`Mapped[int] = mapped_column(BigInteger, autoincrement=True)` 作为二级排序)或用数据库 `func.now()` 服务端时间且测试间隔足够。**为可靠,实现 MemoryStore.write 时显式 `import asyncio; await asyncio.sleep(0)` 不够**;正解:db.py 的 MemoryRow 增加 `seq: Mapped[int] = mapped_column(BigInteger, Identity(), nullable=False)`,`last` 按 `seq.desc()` 排序。请在本任务一并给 MemoryRow 加 `seq` Identity 列并改排序,更新 Task 1 的列断言不必动(`seq` 是额外列)。

- [ ] **Step 4: 跑测试确认通过**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_memory.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/memory packages/ai-employee/src/anvil_ai_employee/db.py packages/ai-employee/tests/test_memory.py
git commit -m "feat(ai-employee): MemoryStore — report_marker long-term memory (monotonic seq order)"
```

---

## Task 6: 异步桥 + 周报员工具

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/asyncbridge.py`
- Create: `packages/ai-employee/src/anvil_ai_employee/tools.py`
- Test: `packages/ai-employee/tests/test_tools.py`

设计:`EmployeeContext` 持有 `session_factory`、`employee` 名、`current_job_id`、惰性构建的 `Retriever`(dense 模式)。四个工具用 P3 `@tool` 装饰器(同步 fn),内部用 `block_on(coro)` 跑异步 DB/检索。`submit_report` 是终止工具:把 markdown 经 queue.complete 写进 job.result + 写 report_marker 记忆。

- [ ] **Step 1: 写 asyncbridge.py(同步桥,复用 M6 模式)**

```python
"""Run an async coroutine to completion from inside a synchronous tool, even when an
outer event loop is already running. Same trick code-agent M6 uses for the summarizer:
hand the coroutine to a worker thread that owns its own asyncio.run."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine
import asyncio


def block_on(coro: Coroutine[Any, Any, Any]) -> Any:
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()
```

- [ ] **Step 2: 写失败测试 test_tools.py**

```python
import json
from datetime import UTC, datetime, timedelta

import pytest

from anvil_ai_employee.db import DocumentRow  # re-exported from anvil_kb via db? see note
from anvil_kb.db import DocumentRow as KbDocumentRow
from anvil_ai_employee.scheduler.queue import claim_one, enqueue
from anvil_ai_employee.tools import EmployeeContext, build_employee_tools

pytestmark = pytest.mark.asyncio


async def _add_doc(session_factory, *, title, source, content, created_at):
    async with session_factory() as s:
        async with s.begin():
            s.add(KbDocumentRow(title=title, source_name=source, content=content, created_at=created_at))


async def _kb_tables(engine):
    # kb_documents lives in anvil_kb metadata; ensure it exists for these tests
    from anvil_kb.db import Base as KbBase
    async with engine.begin() as conn:
        await conn.run_sync(KbBase.metadata.create_all)


async def test_kb_recent_filters_by_since(engine, session_factory):
    await _kb_tables(engine)
    base = datetime(2026, 6, 1, tzinfo=UTC)
    await _add_doc(session_factory, title="old", source="o.md", content="old body", created_at=base)
    await _add_doc(session_factory, title="new", source="n.md", content="new body", created_at=base + timedelta(days=5))
    ctx = EmployeeContext(session_factory=session_factory, employee="kb_reporter", job_id=None)
    from anvil_code_agent.tools.base import ToolContext
    tc = ToolContext(workdir="/tmp")
    tools = {t.name: t for t in build_employee_tools(ctx)}
    res = tools["kb_recent"]({"since_iso": (base + timedelta(days=1)).isoformat()}, tc)
    assert res.ok
    assert "new" in res.content and "old" not in res.content


async def test_recall_marker_empty(engine, session_factory):
    ctx = EmployeeContext(session_factory=session_factory, employee="kb_reporter", job_id=None)
    from anvil_code_agent.tools.base import ToolContext
    tools = {t.name: t for t in build_employee_tools(ctx)}
    res = tools["recall_marker"]({}, ToolContext(workdir="/tmp"))
    assert res.ok and ("从未" in res.content or "never" in res.content.lower())


async def test_submit_report_writes_result_and_marker(engine, session_factory):
    from anvil_code_agent.tools.base import ToolContext
    job_id = await enqueue(session_factory, skill="kb_digest", payload={})
    claimed = await claim_one(session_factory, worker_id="w1")
    ctx = EmployeeContext(session_factory=session_factory, employee="kb_reporter", job_id=claimed.id)
    tools = {t.name: t for t in build_employee_tools(ctx)}
    res = tools["submit_report"](
        {"markdown": "# 周报\n本期 1 篇", "covered_until_iso": "2026-06-08T00:00:00+00:00"},
        ToolContext(workdir="/tmp"),
    )
    assert res.ok
    # job result set
    from sqlalchemy import select
    from anvil_ai_employee.db import JobRow
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one()
        assert row.status == "done" and "周报" in row.result
    # marker written
    from anvil_ai_employee.memory.store import MemoryStore
    last = await MemoryStore(session_factory).last(employee="kb_reporter", kind="report_marker")
    assert last is not None and "2026-06-08" in last
```

> 注意:测试用到 kb_documents 表,需在 conftest 的 engine fixture 里**同时建 anvil_kb 的表**。请在 Task 6 顺手把 conftest 的 engine fixture 增加 `from anvil_kb.db import Base as KbBase; await conn.run_sync(KbBase.metadata.create_all)`(drop 同理),并删除 test_tools.py 里临时的 `_kb_tables` helper、改为依赖 fixture。`DocumentRow` 不必从 ai_employee.db 再导出;直接用 `anvil_kb.db.DocumentRow`。

- [ ] **Step 3: 跑测试确认失败**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_tools.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 4: 写 tools.py**

```python
"""The KB reporter's ACI. P3 @tool protocol (sync fn); async DB/retrieval bridged via
block_on. Tools capture an EmployeeContext at registry-build time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_code_agent.tools.base import Tool, ToolContext, ToolResult, tool
from anvil_kb.db import DocumentRow

from anvil_ai_employee.asyncbridge import block_on
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.scheduler.queue import complete


@dataclass
class EmployeeContext:
    session_factory: async_sessionmaker[AsyncSession]
    employee: str
    job_id: Any  # uuid.UUID | None
    _retriever: Any = None  # lazily built dense Retriever


def _get_retriever(ctx: EmployeeContext):
    if ctx._retriever is None:
        from anvil_kb.embed import FastEmbedEmbedder
        from anvil_kb.retrieve.retriever import Retriever
        from anvil_kb.store.pg import PgVectorStore

        store = PgVectorStore(ctx.session_factory)
        ctx._retriever = Retriever(FastEmbedEmbedder(), store, mode="dense")
    return ctx._retriever


def build_employee_tools(ctx: EmployeeContext) -> list[Tool]:
    async def _recent(since: datetime) -> list[DocumentRow]:
        async with ctx.session_factory() as s:
            return list(
                (
                    await s.execute(
                        select(DocumentRow)
                        .where(DocumentRow.created_at > since)
                        .order_by(DocumentRow.created_at)
                    )
                ).scalars().all()
            )

    @tool(
        name="recall_marker",
        description="回忆上次周报覆盖到的时间点(ISO)。没有则提示从未报告。先调它确定起点。",
        params={}, required=[],
    )
    def recall_marker(args: dict, tc: ToolContext) -> ToolResult:
        last = block_on(MemoryStore(ctx.session_factory).last(
            employee=ctx.employee, kind="report_marker"))
        if not last:
            return ToolResult(content="从未报告过;请覆盖最近 7 天的新增文档。", ok=True)
        return ToolResult(content=f"上次报告标记: {last}", ok=True)

    @tool(
        name="kb_recent",
        description="列出 since_iso 之后新入库的知识库文档(title/source/created_at + 内容预览)。",
        params={"since_iso": {"type": "string", "description": "ISO8601 起始时间"}},
        required=["since_iso"],
    )
    def kb_recent(args: dict, tc: ToolContext) -> ToolResult:
        since = datetime.fromisoformat(args["since_iso"])
        docs = block_on(_recent(since))
        if not docs:
            return ToolResult(content="自该时间点起无新增文档。", ok=True)
        lines = []
        for d in docs:
            preview = (d.content or "")[:200].replace("\n", " ")
            lines.append(f"- [{d.created_at.isoformat()}] {d.title} (source={d.source_name})\n  {preview}")
        return ToolResult(content="新增文档:\n" + "\n".join(lines), ok=True)

    @tool(
        name="kb_search",
        description="对某主题在知识库做语义检索(dense),返回 top-k 片段。用于深入某条目。",
        params={
            "query": {"type": "string"},
            "k": {"type": "integer", "description": "默认 5"},
        },
        required=["query"],
    )
    def kb_search(args: dict, tc: ToolContext) -> ToolResult:
        k = int(args.get("k", 5))
        scored = block_on(_get_retriever(ctx).retrieve(args["query"], k=k))
        if not scored:
            return ToolResult(content="无检索结果。", ok=True)
        out = "\n---\n".join(f"score={s.score:.3f}\n{s.chunk.content}" for s in scored)
        return ToolResult(content=out, ok=True)

    @tool(
        name="submit_report",
        description="提交最终周报 markdown。covered_until_iso 取你这次见过的最大文档时间。提交即完成。",
        params={
            "markdown": {"type": "string"},
            "covered_until_iso": {"type": "string"},
        },
        required=["markdown", "covered_until_iso"],
    )
    def submit_report(args: dict, tc: ToolContext) -> ToolResult:
        markdown = args["markdown"]
        covered = args["covered_until_iso"]

        async def _persist() -> None:
            if ctx.job_id is not None:
                await complete(ctx.session_factory, ctx.job_id, result=markdown)
            await MemoryStore(ctx.session_factory).write(
                employee=ctx.employee, kind="report_marker",
                content=json.dumps({"covered_until": covered, "summary_head": markdown[:120]},
                                   ensure_ascii=False),
            )

        block_on(_persist())
        return ToolResult(content="报告已提交。", ok=True)

    return [recall_marker, kb_recent, kb_search, submit_report]
```

- [ ] **Step 5: 更新 conftest engine fixture 建 kb 表**

在 `conftest.py` 的 engine fixture 内,drop/create 既有 `Base.metadata` 之外,**也对 `anvil_kb.db.Base.metadata` 做 drop_all/create_all**(kb_documents 等)。这样 tools 测试有表可用。

- [ ] **Step 6: 跑测试确认通过**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_tools.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/asyncbridge.py packages/ai-employee/src/anvil_ai_employee/tools.py packages/ai-employee/tests/test_tools.py packages/ai-employee/tests/conftest.py
git commit -m "feat(ai-employee): KB reporter tools (recall/kb_recent/kb_search/submit) + async bridge"
```

---

## Task 7: 技能 persona + build_registry

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/skills/__init__.py`(空)
- Create: `packages/ai-employee/src/anvil_ai_employee/skills/kb_digest.py`
- Test: `packages/ai-employee/tests/test_skill.py`

- [ ] **Step 1: 写失败测试 test_skill.py**

```python
from anvil_ai_employee.skills.kb_digest import PERSONA, build_registry
from anvil_ai_employee.tools import EmployeeContext


def test_persona_mentions_steps():
    assert "recall_marker" in PERSONA
    assert "submit_report" in PERSONA


def test_build_registry_has_four_tools():
    ctx = EmployeeContext(session_factory=None, employee="kb_reporter", job_id=None)
    reg = build_registry(ctx)
    names = {sch["function"]["name"] for sch in reg.schemas()}
    assert names == {"recall_marker", "kb_recent", "kb_search", "submit_report"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/ai-employee/tests/test_skill.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 kb_digest.py**

```python
"""KB reporter skill: persona prompt + registry assembly. A skill = system prompt +
the tool set the agent gets. M2 will version these as markdown skill files."""

from __future__ import annotations

from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.tools import EmployeeContext, build_employee_tools

PERSONA = """你是「知识库周报员」。你的任务是产出一份知识库新增内容的结构化中文周报。

严格按步骤:
1. 调 `recall_marker` 拿到上次报告覆盖到的时间点。
2. 调 `kb_recent(since_iso=<上次时间点>)` 列出此后新入库的文档。
3. 若某主题值得展开,调 `kb_search(query=...)` 深读相关片段。
4. 写一份结构化中文摘要:按主题分组,每条标注来源 source。
5. 调 `submit_report(markdown=<你的周报>, covered_until_iso=<你这次见过的最大文档时间>)` 提交。
   - 若 `kb_recent` 显示无新增,提交一句"本期无新增"并把 covered_until_iso 设为当前任务时间。
提交后即完成,不要再调用其他工具。"""


def build_registry(ctx: EmployeeContext) -> ToolRegistry:
    return ToolRegistry(build_employee_tools(ctx))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/ai-employee/tests/test_skill.py -q`
Expected: PASS(注:build_employee_tools 不在构造期碰 session_factory,故 None 可通过)

- [ ] **Step 5: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/skills packages/ai-employee/tests/test_skill.py
git commit -m "feat(ai-employee): kb_digest skill — persona + registry"
```

---

## Task 8: Worker(run_once)

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/worker.py`
- Test: `packages/ai-employee/tests/test_worker.py`

设计:`run_once(session_factory, *, model, worker_id, max_steps=12)`:claim 一个 job → 若 None 返回 False → 按 skill 取 registry → 跑 P3 `run` → 若 agent 没调 submit_report(job 仍非 done)则用 agent 末条消息 fallback complete → 异常走 fail → 返回 True。包在 obs span。测试用 respx mock gateway 录一段"调 recall_marker → 调 submit_report → 结束"的工具序列。

- [ ] **Step 1: 写失败测试 test_worker.py(respx mock 录制工具序列)**

```python
import json

import httpx
import pytest
import respx

from anvil_ai_employee.scheduler.queue import enqueue
from anvil_ai_employee.worker import run_once

pytestmark = pytest.mark.asyncio

CHAT_URL = "https://api.deepseek.com/chat/completions"  # adjust to gateway base if needed


def _assistant_tool_call(tool_id, name, arguments):
    return {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": tool_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def _assistant_done():
    return {"choices": [{"message": {"role": "assistant", "content": "完成"}}]}


@respx.mock
async def test_run_once_executes_job_to_done(engine, session_factory, monkeypatch):
    # ensure kb tables exist via engine fixture; seed a doc so kb_recent has data
    from datetime import UTC, datetime
    from anvil_kb.db import DocumentRow
    async with session_factory() as s:
        async with s.begin():
            s.add(DocumentRow(title="新政策", source_name="p.md", content="保单新增条款……",
                              created_at=datetime(2026, 6, 8, tzinfo=UTC)))

    route = respx.post(CHAT_URL)
    route.side_effect = [
        httpx.Response(200, json=_assistant_tool_call("c1", "recall_marker", {})),
        httpx.Response(200, json=_assistant_tool_call("c2", "kb_recent",
                        {"since_iso": "2026-06-01T00:00:00+00:00"})),
        httpx.Response(200, json=_assistant_tool_call("c3", "submit_report",
                        {"markdown": "# 周报\n- 新政策 (p.md)", "covered_until_iso": "2026-06-08T00:00:00+00:00"})),
        httpx.Response(200, json=_assistant_done()),
    ]

    job_id = await enqueue(session_factory, skill="kb_digest", payload={})
    ok = await run_once(session_factory, model="deepseek-chat", worker_id="w1", max_steps=12)
    assert ok is True

    from sqlalchemy import select
    from anvil_ai_employee.db import JobRow
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one()
        assert row.status == "done"
        assert "周报" in row.result


@respx.mock
async def test_run_once_no_job_returns_false(engine, session_factory):
    assert await run_once(session_factory, model="deepseek-chat", worker_id="w1") is False
```

> 实现者注意:gateway 的真实 base URL/鉴权头以 `anvil_gateway` 现有 respx 测试为准(参考 `packages/code-agent/tests` 里 loop 的 mock 写法),`CHAT_URL` 按那边对齐。submit_report 在工具内部已 complete 该 job,故 worker 末尾若发现 job 已 done 就不再 fallback。

- [ ] **Step 2: 跑测试确认失败**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_worker.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 worker.py**

```python
"""Worker: claim a job, run the matching skill's agent loop (reusing P3 harness),
persist outcome. submit_report already marks the job done; worker handles the rest."""

from __future__ import annotations

import tempfile

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_code_agent.harness.loop import run as agent_run
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext
from anvil_obs import span

from anvil_ai_employee.db import JobRow
from anvil_ai_employee.scheduler.queue import claim_one, complete, fail
from anvil_ai_employee.skills import kb_digest
from anvil_ai_employee.tools import EmployeeContext

SKILLS = {"kb_digest": (kb_digest.PERSONA, kb_digest.build_registry)}
TASK_PROMPT = "现在开始产出本期知识库周报。"


async def _job_status(session_factory, job_id) -> str:
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one()
        return row.status


async def run_once(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model: str,
    worker_id: str,
    max_steps: int = 12,
) -> bool:
    """Claim and run one job. Returns False if the queue was empty."""
    job = await claim_one(session_factory, worker_id=worker_id)
    if job is None:
        return False
    with span("ai_employee.job", skill=job.skill, worker=worker_id):
        try:
            if job.skill not in SKILLS:
                await fail(session_factory, job.id, error=f"unknown skill: {job.skill}")
                return True
            persona, build_registry = SKILLS[job.skill]
            ctx = EmployeeContext(
                session_factory=session_factory, employee="kb_reporter", job_id=job.id
            )
            registry = build_registry(ctx)
            with tempfile.TemporaryDirectory() as workdir:
                state = AgentState.new(
                    system=persona, task=TASK_PROMPT, workdir=workdir, max_steps=max_steps
                )
                tc = ToolContext(workdir=workdir)
                await agent_run(state, model, registry, tc)
            # submit_report already completed the job; fallback if the agent never submitted
            if await _job_status(session_factory, job.id) != "done":
                await fail(
                    session_factory, job.id,
                    error="agent finished without calling submit_report",
                )
        except Exception as e:  # noqa: BLE001 — isolate one job's failure
            await fail(session_factory, job.id, error=f"{type(e).__name__}: {e}")
    return True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_worker.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/worker.py packages/ai-employee/tests/test_worker.py
git commit -m "feat(ai-employee): worker run_once — claim job, run P3 loop, persist outcome"
```

---

## Task 9: CLI

**Files:**
- Create: `packages/ai-employee/src/anvil_ai_employee/cli.py`
- Test: `packages/ai-employee/tests/test_cli.py`(注意文件名避免与他包 test_cli 冲突:命名 `test_ai_employee_cli.py`)

设计:argparse,子命令 `add-schedule` / `tick` / `work` / `run-now` / `report`。`tick`/`work` 支持 `--loop`(轮询睡眠)。`add-schedule` 计算首个 next_run_at = `croniter(cron, now).get_next()`。所有命令读 `ANVIL_DATABASE_URL`。`main()` 为 entry point。

- [ ] **Step 1: 写失败测试 test_ai_employee_cli.py(测纯函数,不起循环)**

```python
import pytest
from datetime import UTC, datetime

from anvil_ai_employee.cli import add_schedule, run_now, show_report

pytestmark = pytest.mark.asyncio


async def test_add_schedule_inserts_with_next_run(session_factory):
    now = datetime(2026, 6, 9, 8, 0, tzinfo=UTC)
    sid = await add_schedule(session_factory, name="周报", cron="0 9 * * 1",
                             skill="kb_digest", now=now)
    from sqlalchemy import select
    from anvil_ai_employee.db import ScheduleRow
    async with session_factory() as s:
        row = (await s.execute(select(ScheduleRow).where(ScheduleRow.id == sid))).scalar_one()
        assert row.name == "周报" and row.next_run_at > now


async def test_run_now_enqueues(session_factory):
    jid = await run_now(session_factory, skill="kb_digest")
    from sqlalchemy import select
    from anvil_ai_employee.db import JobRow
    async with session_factory() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == jid))).scalar_one()
        assert row.status == "pending" and row.skill == "kb_digest"


async def test_show_report_for_missing_job(session_factory):
    import uuid
    out = await show_report(session_factory, job_id=uuid.uuid4())
    assert "未找到" in out or "not found" in out.lower()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_ai_employee_cli.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 写 cli.py**

```python
"""anvil-ai-employee CLI: add-schedule / tick / work / run-now / report.

The cron *ticker* (`tick`) and the *worker* (`work`) are separate processes — system
cron can call `tick`, or run `tick --loop` / `work --loop` as long-running daemons."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import datetime, timezone

from croniter import croniter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import JobRow, ScheduleRow, make_session_factory
from anvil_ai_employee.scheduler.queue import enqueue
from anvil_ai_employee.scheduler.trigger import CronTrigger
from anvil_ai_employee.worker import run_once


async def add_schedule(
    sf: async_sessionmaker[AsyncSession], *, name: str, cron: str, skill: str,
    now: datetime | None = None,
) -> uuid.UUID:
    now = now or datetime.now(timezone.utc)
    next_run = croniter(cron, now).get_next(datetime)
    sid = uuid.uuid4()
    async with sf() as s:
        async with s.begin():
            s.add(ScheduleRow(id=sid, name=name, cron_expr=cron, skill=skill,
                              payload={}, next_run_at=next_run, enabled=True))
    return sid


async def run_now(sf: async_sessionmaker[AsyncSession], *, skill: str) -> uuid.UUID:
    return await enqueue(sf, skill=skill, payload={})


async def show_report(sf: async_sessionmaker[AsyncSession], *, job_id: uuid.UUID) -> str:
    async with sf() as s:
        row = (await s.execute(select(JobRow).where(JobRow.id == job_id))).scalar_one_or_none()
    if row is None:
        return f"未找到 job {job_id}"
    return f"[{row.status}] skill={row.skill}\n{row.result or row.error or '(无输出)'}"


async def _tick_loop(sf, *, once: bool, interval: float = 60.0) -> None:
    trig = CronTrigger(sf)
    while True:
        n = await trig.due(datetime.now(timezone.utc))
        print(f"tick: enqueued {n} job(s)")
        if once:
            return
        await asyncio.sleep(interval)


async def _work_loop(sf, *, model: str, worker_id: str, once: bool, interval: float = 5.0) -> None:
    while True:
        did = await run_once(sf, model=model, worker_id=worker_id)
        if not did:
            if once:
                return
            await asyncio.sleep(interval)


def main() -> None:
    p = argparse.ArgumentParser(prog="anvil-ai-employee")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add-schedule")
    a.add_argument("--name", required=True)
    a.add_argument("--cron", required=True)
    a.add_argument("--skill", default="kb_digest")

    t = sub.add_parser("tick")
    t.add_argument("--loop", action="store_true")

    w = sub.add_parser("work")
    w.add_argument("--model", default="deepseek-chat")
    w.add_argument("--worker-id", default="w1")
    w.add_argument("--loop", action="store_true")

    r = sub.add_parser("run-now")
    r.add_argument("--skill", default="kb_digest")

    rp = sub.add_parser("report")
    rp.add_argument("--job", required=True)

    args = p.parse_args()
    sf = make_session_factory()  # reads ANVIL_DATABASE_URL

    if args.cmd == "add-schedule":
        sid = asyncio.run(add_schedule(sf, name=args.name, cron=args.cron, skill=args.skill))
        print(f"schedule {sid} created")
    elif args.cmd == "tick":
        asyncio.run(_tick_loop(sf, once=not args.loop))
    elif args.cmd == "work":
        asyncio.run(_work_loop(sf, model=args.model, worker_id=args.worker_id, once=not args.loop))
    elif args.cmd == "run-now":
        jid = asyncio.run(run_now(sf, skill=args.skill))
        print(f"job {jid} enqueued")
    elif args.cmd == "report":
        print(asyncio.run(show_report(sf, job_id=uuid.UUID(args.job))))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `ANVIL_DATABASE_URL=... uv run pytest packages/ai-employee/tests/test_ai_employee_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/ai-employee/src/anvil_ai_employee/cli.py packages/ai-employee/tests/test_ai_employee_cli.py
git commit -m "feat(ai-employee): CLI — add-schedule/tick/work/run-now/report"
```

---

## Task 10: 全仓校验 + example README + 收尾

**Files:**
- Create: `examples/07-ai-employee/README.md`
- Test: 全仓 pytest + ruff

- [ ] **Step 1: 写 examples/07-ai-employee/README.md**

内容覆盖:产品定位(P4 集大成 M1)、架构图文字版(cron→PG 队列→worker→P3 循环→记忆标记)、复用了哪些既有产品(P1 KB / P3 harness / gateway / obs)、跑法:

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
# 1. 灌点知识库内容(复用 P1)
uv run anvil-kb ingest packages/kb/golden/corpus/*.md
# 2. 立即派一个周报任务
uv run anvil-ai-employee run-now --skill kb_digest
# 3. 起 worker 跑一次(需 DEEPSEEK_API_KEY)
uv run anvil-ai-employee work
# 4. 看周报
uv run anvil-ai-employee report --job <id>
# 定时:每周一 9 点
uv run anvil-ai-employee add-schedule --name 周报 --cron "0 9 * * 1"
uv run anvil-ai-employee tick --loop &   # ticker
uv run anvil-ai-employee work --loop &   # worker
```

明确标注 M1 范围与 M2~M5 留待(三层记忆/HITL/MCP/多员工)。

- [ ] **Step 2: 全仓 ruff(CI 等价)**

Run: `uv run ruff check .`
Expected: 全绿。**有问题就地修并 `git add` 提交**(CA-M5 教训:本地分包过、CI 全仓炸,且修复必须提交)。

- [ ] **Step 3: 全仓测试(non-live)**

Run: `ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test uv run pytest -m "not live" -q`
Expected: 全绿,包含 ai-employee 全部用例,且**不破坏其他包**(尤其 test_cli 命名不冲突、无表名/fixture 污染)。

- [ ] **Step 4: Commit + 更新 CLAUDE.md**

在 anvil 根 `CLAUDE.md` 增一节 `## anvil-ai-employee (packages/ai-employee)` 简述本包(对照 anvil-code-agent 那节的体例:模块、命令、测试、复用)。

```bash
git add examples/07-ai-employee/README.md CLAUDE.md
git commit -m "docs(ai-employee): example README + CLAUDE.md section for P4-M1"
```

- [ ] **Step 5: 推分支 + 开 PR**

```bash
git push -u origin feat/ai-employee-m1
gh pr create --title "feat: P4-M1 AI 员工「知识库周报员」" --body "<总结 M1 范围与复用>"
```

---

## Self-Review 检查

- **Spec 覆盖**:三表(Task1)、PG 队列(Task3)、CronTrigger(Task4)、最小记忆(Task5)、四工具+异步桥(Task6)、技能(Task7)、worker 复用 P3(Task8)、CLI 五命令(Task9)、example+校验(Task10)——spec 各节均有对应任务。✅
- **占位扫描**:无 TBD;每个 code step 给了完整代码。Task4/5/6 的"实现者注意"是明确的择一/补强指令,非占位。
- **类型一致**:`enqueue/claim_one/complete/fail` 跨任务签名一致;`EmployeeContext` 字段在 Task6 定义、Task7/8 复用一致;`build_registry(ctx)`/`build_employee_tools(ctx)` 命名贯穿。
- **CI 雷区**:Task9 测试文件命名 `test_ai_employee_cli.py` 防与 kb/code-agent 的 test_cli 冲突;Task10 强制全仓 ruff+pytest 并提交修复(CA-M5 教训)。
