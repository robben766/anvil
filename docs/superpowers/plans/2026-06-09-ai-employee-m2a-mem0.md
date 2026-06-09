# P4-M2a mem0 抽取式记忆 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development。Steps 用 `- [ ]`。

**Goal:** 给 AI 员工装上 mem0 哲学的托管式长期记忆 + 对话型助理 CLI;每轮对话后 LLM 抽取事实→近邻比对 ADD/UPDATE/DELETE/NOOP→存向量;跨轮/跨会话向量召回注入。

**Architecture:** 见 spec `docs/superpowers/specs/2026-06-09-ai-employee-m2a-memory-mem0-design.md`。新建 memory/{vectorstore,strategy,mem0}.py + sessions.py + chat.py + eval/memory/;扩 db.py(ae_memories+embedding、ae_sessions)与 MemoryStore;给 code-agent AgentState 加 resume/from_messages reducer。

**Tech Stack:** Python 3.12 / SQLAlchemy 异步 + pgvector Vector(512) / 复用 P3 harness、guard.structured_chat、kb FastEmbedEmbedder、asyncbridge。PG@5434,无 SQLite。

---

## 关键既有接口(照此调用,勿臆造)

- `anvil_code_agent.state.AgentState`:frozen dataclass(messages tuple/step/max_steps/workdir/status);`new(*,system,task,workdir,max_steps)`、`append`、`advance`、`finish`、`replace`(用 dataclasses.replace)。本计划给它加 `resume`/`from_messages`。
- `anvil_code_agent.harness.loop.run(state, model, registry, ctx, *, policy=auto_approve, token_budget=None, summarizer=None)` 异步,驱动到 done/exhausted。`tools.base`:`ToolRegistry`/`ToolContext(workdir,...)`/`@tool`。
- `anvil_guard.structured_chat(model, messages, *, schema, max_retries=1, **chat_kwargs)` → dict;走 `response_format={"type":"json_object"}`,**messages 文本必须含字面 "json"**;只校验 `schema["required"]` 键存在,不校验枚举值。
- `anvil_kb.embed.FastEmbedEmbedder()`:`embed_texts(list[str])->list[list[float]]`(passage 无前缀)、`embed_query(str)->list[float]`(带中文检索前缀)。`anvil_kb.embed.Embedder` 是 Protocol(可 stub)。
- `anvil_kb.store.pg.PgVectorStore.search` 写法参考:`Row.embedding.cosine_distance(query_vector).label("distance")` → `.order_by("distance").limit(k)`。**不可直接复用**(绑 ChunkRow)。
- `pgvector.sqlalchemy.Vector`;`anvil_kb.db.EMBEDDING_DIM == 512`。
- M1 现有:`anvil_ai_employee.db`(Base/ScheduleRow/JobRow/MemoryRow<id,employee,kind,content,seq,created_at>/make_engine/make_session_factory)、`memory/store.py`(MemoryStore.write/last)、`tools.py`(EmployeeContext)、`asyncbridge.block_on`、conftest(engine 建 ae+kb 表 + autouse `_gateway_env`)。

---

## File Structure

- Modify `packages/code-agent/src/anvil_code_agent/state.py` — 加 resume/from_messages(加法)
- Modify `packages/ai-employee/src/anvil_ai_employee/db.py` — ae_memories +embedding;新增 SessionRow
- Modify `packages/ai-employee/src/anvil_ai_employee/memory/store.py` — MemoryStore 扩 insert/update/delete/list_facts
- Create `packages/ai-employee/src/anvil_ai_employee/memory/vectorstore.py` — MemoryVectorStore.knn
- Create `packages/ai-employee/src/anvil_ai_employee/memory/strategy.py` — MemoryStrategy 协议 + NoMemoryStrategy
- Create `packages/ai-employee/src/anvil_ai_employee/memory/mem0.py` — Mem0Strategy
- Create `packages/ai-employee/src/anvil_ai_employee/sessions.py` — SessionStore
- Create `packages/ai-employee/src/anvil_ai_employee/chat.py` — run_one_turn + chat_repl
- Modify `packages/ai-employee/src/anvil_ai_employee/cli.py` — 加 chat 子命令
- Create `packages/ai-employee/src/anvil_ai_employee/eval/memory/...` — golden fixture + 评测
- 对应 tests/*

---

## Task 1: AgentState 加 resume / from_messages reducer

**Files:** Modify `packages/code-agent/src/anvil_code_agent/state.py`;Test `packages/code-agent/tests/test_state_resume.py`

- [ ] **Step 1: 失败测试**

```python
from anvil_code_agent.state import AgentState


def _state_done():
    s = AgentState.new(system="sys", task="hi", workdir="/tmp", max_steps=5)
    return s.append({"role": "assistant", "content": "done"}).advance().finish("done")


def test_resume_rearms_finished_state():
    s = _state_done()
    assert s.status == "done"
    r = s.resume({"role": "user", "content": "next"})
    assert r.status == "running"
    assert r.step == 0  # per-turn budget reset
    assert r.messages[-1] == {"role": "user", "content": "next"}
    assert len(r.messages) == len(s.messages) + 1


def test_from_messages_rehydrates():
    msgs = (
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    )
    s = AgentState.from_messages(msgs, workdir="/tmp", max_steps=7)
    assert s.messages == msgs
    assert s.status == "running" and s.step == 0 and s.max_steps == 7
```

- [ ] **Step 2: 跑确认失败** `uv run pytest packages/code-agent/tests/test_state_resume.py -q` → FAIL(AttributeError)

- [ ] **Step 3: 实现(加法,不改既有方法)**

在 `AgentState` 内加:
```python
    def resume(self, user_msg: Message) -> AgentState:
        """Re-arm a finished chat state for another user turn: append the user message,
        reset to running, reset the per-turn step counter (max_steps is per-turn)."""
        return replace(self, messages=self.messages + (user_msg,), status="running", step=0)

    @classmethod
    def from_messages(
        cls, messages: tuple[Message, ...], *, workdir: str, max_steps: int,
        status: Status = "running",
    ) -> AgentState:
        """Rehydrate a state from a persisted message tuple (chat session resume)."""
        return cls(messages=messages, step=0, max_steps=max_steps, workdir=workdir, status=status)
```

- [ ] **Step 4: 跑确认通过**;并跑 `uv run pytest packages/code-agent -q` 确认没破坏 P3 既有(resume/from_messages 是加法)。

- [ ] **Step 5: ruff** `uv run ruff check packages/code-agent --fix && uv run ruff check packages/code-agent`

- [ ] **Step 6: Commit**
```bash
git add packages/code-agent/src/anvil_code_agent/state.py packages/code-agent/tests/test_state_resume.py
git commit -m "feat(code-agent): AgentState.resume/from_messages reducers for multi-turn chat"
```

---

## Task 2: db.py — ae_memories +embedding,新增 SessionRow

**Files:** Modify `packages/ai-employee/src/anvil_ai_employee/db.py`;Test `packages/ai-employee/tests/test_ai_employee_db.py`(M1 已有,追加断言)

- [ ] **Step 1: 追加失败断言**(在 M1 的 test_ai_employee_db.py 里加)

```python
def test_memory_has_embedding_and_session_row():
    from anvil_ai_employee.db import MemoryRow, SessionRow
    assert "embedding" in MemoryRow.__table__.columns.keys()
    scols = SessionRow.__table__.columns.keys()
    assert {"id", "employee", "messages", "status", "created_at", "updated_at"} <= set(scols)
    assert SessionRow.__tablename__ == "ae_sessions"
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现**

`db.py`:`MemoryRow` 加
```python
    from pgvector.sqlalchemy import Vector  # top-level import
    embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)
```
(把 import 提到文件顶部。)新增:
```python
class SessionRow(Base):
    __tablename__ = "ae_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee: Mapped[str] = mapped_column(Text, nullable=False)
    messages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
```
注释写明不变量:`kind in {fact,archival}` ⇒ embedding NOT NULL;`{core,report_marker}` ⇒ NULL(app 层保证)。

- [ ] **Step 4: 跑确认通过**(需 PG)`ANVIL_DATABASE_URL=...anvil_test uv run pytest packages/ai-employee/tests/test_ai_employee_db.py packages/ai-employee/tests/test_schema_create.py -q`

- [ ] **Step 5: ruff** `uv run ruff check packages/ai-employee --fix && uv run ruff check packages/ai-employee`

- [ ] **Step 6: Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/db.py packages/ai-employee/tests/test_ai_employee_db.py
git commit -m "feat(ai-employee): ae_memories.embedding + ae_sessions table"
```

---

## Task 3: MemoryStore 扩 API + MemoryVectorStore

**Files:** Modify `memory/store.py`;Create `memory/vectorstore.py`;Test `tests/test_memory_store_v2.py`

- [ ] **Step 1: 失败测试**

```python
import pytest
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.memory.vectorstore import MemoryVectorStore

pytestmark = pytest.mark.asyncio


def _vec(seed: float) -> list[float]:
    return [seed] * 512


async def test_insert_update_delete_list(session_factory):
    store = MemoryStore(session_factory)
    mid = await store.insert(employee="u1", kind="fact", content="住在北京", embedding=_vec(0.1))
    facts = await store.list_facts(employee="u1")
    assert len(facts) == 1 and facts[0].content == "住在北京"
    await store.update(mid, content="住在上海", embedding=_vec(0.2))
    facts = await store.list_facts(employee="u1")
    assert facts[0].content == "住在上海"
    await store.delete(mid)
    assert await store.list_facts(employee="u1") == []


async def test_last_still_works_for_report_marker(session_factory):
    store = MemoryStore(session_factory)
    await store.insert(employee="u1", kind="report_marker", content="m1")
    await store.insert(employee="u1", kind="report_marker", content="m2")
    assert await store.last(employee="u1", kind="report_marker") == "m2"


async def test_vector_knn_filters_employee_and_kind(session_factory):
    store = MemoryStore(session_factory)
    await store.insert(employee="u1", kind="fact", content="北京", embedding=_vec(0.9))
    await store.insert(employee="u1", kind="fact", content="猫", embedding=_vec(0.1))
    await store.insert(employee="u2", kind="fact", content="别人", embedding=_vec(0.9))
    vs = MemoryVectorStore(session_factory)
    hits = await vs.knn(employee="u1", kinds=["fact"], query_vec=_vec(0.9), k=5)
    contents = [row.content for row, score in hits]
    assert "北京" in contents and "别人" not in contents  # employee isolation
    assert hits[0][0].content == "北京"  # nearest first
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现 store.py(扩 MemoryStore,保留 write/last)**

```python
import uuid
from sqlalchemy import delete as sa_delete
from sqlalchemy import select, update as sa_update
# ... MemoryStore 内新增:
    async def insert(self, *, employee, kind, content, embedding=None) -> uuid.UUID:
        mid = uuid.uuid4()
        async with self._sf() as s:
            async with s.begin():
                s.add(MemoryRow(id=mid, employee=employee, kind=kind,
                                content=content, embedding=embedding))
        return mid

    async def update(self, mem_id, *, content, embedding=None) -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(sa_update(MemoryRow).where(MemoryRow.id == mem_id)
                                .values(content=content, embedding=embedding))

    async def delete(self, mem_id) -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(sa_delete(MemoryRow).where(MemoryRow.id == mem_id))

    async def list_facts(self, *, employee, kind="fact") -> list[MemoryRow]:
        async with self._sf() as s:
            return list((await s.execute(
                select(MemoryRow).where(MemoryRow.employee == employee)
                .where(MemoryRow.kind == kind).order_by(MemoryRow.seq))).scalars().all())
```
(保留 M1 的 `write`/`last` 不动;`write` 可保留或让其内部转调 insert,二选一不破坏 M1 测试。)

- [ ] **Step 4: 实现 vectorstore.py**

```python
"""Vector KNN over ae_memories. NOT reusing kb's PgVectorStore — that one binds ChunkRow.
Filters by employee (multi-employee isolation) and kind."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import MemoryRow


class MemoryVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def knn(self, *, employee: str, kinds: list[str], query_vec: list[float], k: int
                  ) -> list[tuple[MemoryRow, float]]:
        async with self._sf() as s:
            rows = (await s.execute(
                select(MemoryRow, MemoryRow.embedding.cosine_distance(query_vec).label("d"))
                .where(MemoryRow.employee == employee)
                .where(MemoryRow.kind.in_(kinds))
                .where(MemoryRow.embedding.is_not(None))
                .order_by("d").limit(k)
            )).all()
        return [(row[0], 1.0 - float(row[1])) for row in rows]
```

- [ ] **Step 5: 跑确认通过**(需 PG)

- [ ] **Step 6: ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/memory/store.py packages/ai-employee/src/anvil_ai_employee/memory/vectorstore.py packages/ai-employee/tests/test_memory_store_v2.py
git commit -m "feat(ai-employee): MemoryStore CRUD + MemoryVectorStore knn (employee/kind filtered)"
```

---

## Task 4: MemoryStrategy 协议 + NoMemoryStrategy

**Files:** Create `memory/strategy.py`;Test `tests/test_strategy.py`

- [ ] **Step 1: 失败测试**

```python
import pytest
from anvil_ai_employee.memory.strategy import NoMemoryStrategy

pytestmark = pytest.mark.asyncio


async def test_no_memory_strategy_is_noop():
    strat = NoMemoryStrategy()
    reg = strat.build_registry(ctx=None)
    assert reg.schemas() == []
    assert await strat.system_prefix("u1", "hi") == ""
    # after_turn must be awaitable no-op
    await strat.after_turn("u1", None, [])
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现**

```python
"""MemoryStrategy: the seam between a chat employee and a memory philosophy.
mem0 (managed) drives memory in the orchestrator (system_prefix recall + after_turn update).
Letta (self-managed, M2b) drives it via agent tools. NoMemoryStrategy is the baseline."""

from __future__ import annotations

from typing import Any, Protocol

from anvil_code_agent.tools.base import ToolRegistry


class MemoryStrategy(Protocol):
    def build_registry(self, ctx: Any) -> ToolRegistry: ...
    async def system_prefix(self, employee: str, user_msg: str) -> str: ...
    async def after_turn(self, employee: str, session: Any, msgs: list[dict]) -> None: ...


class NoMemoryStrategy:
    """Baseline: no recall, no tools, no learning."""

    def build_registry(self, ctx: Any) -> ToolRegistry:
        return ToolRegistry([])

    async def system_prefix(self, employee: str, user_msg: str) -> str:
        return ""

    async def after_turn(self, employee: str, session: Any, msgs: list[dict]) -> None:
        return None
```

- [ ] **Step 4: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/memory/strategy.py packages/ai-employee/tests/test_strategy.py
git commit -m "feat(ai-employee): MemoryStrategy protocol + NoMemoryStrategy baseline"
```

---

## Task 5: Mem0Strategy(novel core,最高风险)

**Files:** Create `memory/mem0.py`;Test `tests/test_mem0_strategy.py`

实现 Mem0Strategy。**抽取/比对走 guard.structured_chat(prompt 含字面 "json");写库 embed_texts、召回 embed_query;reconcile op∈{ADD,UPDATE,DELETE,NOOP} + Python 侧校验非法当 NOOP。**

- [ ] **Step 1: 失败测试(stub Embedder + respx 录 structured_chat,确定性)**

```python
import json
import httpx
import pytest
import respx
from anvil_ai_employee.memory.mem0 import Mem0Strategy
from anvil_ai_employee.memory.store import MemoryStore

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"


class StubEmbedder:
    """Deterministic: vector keyed by whether text mentions 居住/北京/上海 so that
    '住在上海' and '住在北京' are nearest neighbors (the mem0 near-neighbor assumption)."""
    def _vec(self, text):
        base = 0.9 if ("住" in text or "京" in text or "海" in text) else 0.1
        return [base] * 512
    def embed_texts(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)


def _json_resp(obj):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
        "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


@respx.mock
async def test_first_fact_is_added(session_factory):
    # extract → 1 fact ; reconcile → ADD
    respx.post(DS_URL).mock(side_effect=[
        _json_resp({"facts": ["用户住在北京"]}),
        _json_resp({"op": "ADD"}),
    ])
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn("u1", None, [
        {"role": "user", "content": "我住在北京"},
        {"role": "assistant", "content": "好的"}])
    facts = await MemoryStore(session_factory).list_facts(employee="u1")
    assert len(facts) == 1 and "北京" in facts[0].content
    assert facts[0].embedding is not None  # ADD path used embed_texts


@respx.mock
async def test_contradiction_updates_not_double_adds(session_factory):
    store = MemoryStore(session_factory)
    # seed an existing 北京 fact
    emb = StubEmbedder().embed_texts(["用户住在北京"])[0]
    existing = await store.insert(employee="u1", kind="fact", content="用户住在北京", embedding=emb)
    # extract 上海 → reconcile UPDATE target=existing
    respx.post(DS_URL).mock(side_effect=[
        _json_resp({"facts": ["用户住在上海"]}),
        _json_resp({"op": "UPDATE", "target_id": str(existing)}),
    ])
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn("u1", None, [
        {"role": "user", "content": "我搬到上海了"},
        {"role": "assistant", "content": "记住了"}])
    facts = await store.list_facts(employee="u1")
    assert len(facts) == 1 and "上海" in facts[0].content  # updated, not double-added


@respx.mock
async def test_illegal_op_falls_back_to_noop(session_factory):
    respx.post(DS_URL).mock(side_effect=[
        _json_resp({"facts": ["用户喜欢猫"]}),
        _json_resp({"op": "FROBNICATE"}),  # illegal
    ])
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn("u1", None, [
        {"role": "user", "content": "我喜欢猫"},
        {"role": "assistant", "content": "ok"}])
    # NOOP fallback → nothing written
    assert await MemoryStore(session_factory).list_facts(employee="u1") == []


@respx.mock
async def test_system_prefix_recalls(session_factory):
    store = MemoryStore(session_factory)
    emb = StubEmbedder().embed_texts(["用户住在北京"])[0]
    await store.insert(employee="u1", kind="fact", content="用户住在北京", embedding=emb)
    strat = Mem0Strategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    prefix = await strat.system_prefix("u1", "我住哪来着")
    assert "北京" in prefix
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现 mem0.py**

要点:
- `__init__(self, session_factory, *, embedder, model, recall_k=5, neighbor_k=5, window=10)`;`self._store=MemoryStore(sf)`、`self._vs=MemoryVectorStore(sf)`。
- `build_registry(ctx)` → `ToolRegistry([])`。
- `system_prefix(employee, user_msg)`:`qv=embedder.embed_query(user_msg)`;`hits=await self._vs.knn(employee, ["fact"], qv, recall_k)`;无命中返回 `""`;否则返回 `"# 关于用户你已知道:\n" + "\n".join(f"- {row.content}" for row,_ in hits)`。
- `after_turn(employee, session, msgs)`:包 `with span("ai_employee.mem0.after_turn")`:
  1. 抽取上下文 window:`msgs`(当前 pair)即可(session 取最近 N 条留 spiral;M2a 先只用当前 pair + 若 session 提供 messages 则取末 window 条)。
  2. `extract = await structured_chat(model, [{"role":"system","content":EXTRACT_PROMPT},{"role":"user","content":json.dumps(window, ensure_ascii=False)}], schema={"required":["facts"]})`。`EXTRACT_PROMPT` 含字面 "json",要求"抽取关于用户的离散事实,返回 {\"facts\": [...]} 的 json"。
  3. `facts = extract.get("facts") or []`;逐 fact:
     - `nb = await self._vs.knn(employee, ["fact"], embedder.embed_query(fact), neighbor_k)`
     - `dec = await structured_chat(model, [{"role":"system","content":RECONCILE_PROMPT},{"role":"user","content":reconcile_user(fact, nb)}], schema={"required":["op"]})`(prompt 含 "json",列出 neighbors 的 id+content,要求选 op∈ADD/UPDATE/DELETE/NOOP,UPDATE/DELETE 带 target_id)
     - `op=str(dec.get("op","")).upper()`;`tid=dec.get("target_id")`;**校验**:op 不在四枚举 → NOOP;UPDATE/DELETE 但 tid 不在 nb 的 id 串集合 → NOOP(记 obs)。
     - apply:ADD→`store.insert(kind="fact", content=fact, embedding=embedder.embed_texts([fact])[0])`;UPDATE→`store.update(uuid(tid), content=fact, embedding=embedder.embed_texts([fact])[0])`;DELETE→`store.delete(uuid(tid))`;NOOP→pass。
  4. 任何 `StructuredOutputError` → 记 obs 跳过本次更新(不抛)。
- 模块顶部放 `EXTRACT_PROMPT`/`RECONCILE_PROMPT` 常量(中文,含字面 "json")。

- [ ] **Step 4: 跑确认通过**(需 PG;structured_chat 走 respx,conftest autouse gateway fixture 已配)

- [ ] **Step 5: ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/memory/mem0.py packages/ai-employee/tests/test_mem0_strategy.py
git commit -m "feat(ai-employee): Mem0Strategy — extract/reconcile ADD-UPDATE-DELETE-NOOP + recall"
```

---

## Task 6: SessionStore

**Files:** Create `sessions.py`;Test `tests/test_sessions.py`

- [ ] **Step 1: 失败测试**

```python
import pytest
from anvil_ai_employee.sessions import SessionStore

pytestmark = pytest.mark.asyncio


async def test_create_save_load_roundtrip(session_factory):
    ss = SessionStore(session_factory)
    sid = await ss.create(employee="assistant")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    await ss.save(sid, msgs, status="active")
    loaded = await ss.load(sid)
    assert loaded == tuple(msgs)


async def test_load_missing_returns_empty_tuple(session_factory):
    import uuid
    ss = SessionStore(session_factory)
    assert await ss.load(uuid.uuid4()) == ()
```

- [ ] **Step 2: 跑失败 → 实现 sessions.py**

```python
"""Session tier: persist a chat's conversation messages (NOT the per-turn re-injected
system block) so a session can resume across runs. messages are plain dicts (P3 fact)."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import SessionRow


class SessionStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def create(self, *, employee: str) -> uuid.UUID:
        sid = uuid.uuid4()
        async with self._sf() as s:
            async with s.begin():
                s.add(SessionRow(id=sid, employee=employee, messages=[], status="active"))
        return sid

    async def save(self, sid: uuid.UUID, messages: list[dict], *, status: str = "active") -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(sa_update(SessionRow).where(SessionRow.id == sid)
                                .values(messages=messages, status=status))

    async def load(self, sid: uuid.UUID) -> tuple[dict, ...]:
        async with self._sf() as s:
            row = (await s.execute(select(SessionRow).where(SessionRow.id == sid))).scalar_one_or_none()
        return tuple(row.messages) if row else ()
```

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/sessions.py packages/ai-employee/tests/test_sessions.py
git commit -m "feat(ai-employee): SessionStore — persist/resume chat conversation messages"
```

---

## Task 7: chat — run_one_turn + chat_repl

**Files:** Create `chat.py`;Test `tests/test_chat.py`

run_one_turn 串起 prepare→resume→run→after_turn→save。

- [ ] **Step 1: 失败测试(respx mock,NoMemoryStrategy 先验证穿线,再验证 strategy 钩子被调)**

```python
import json
import httpx
import pytest
import respx
from anvil_ai_employee.chat import run_one_turn
from anvil_ai_employee.memory.strategy import NoMemoryStrategy

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _text(t):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": t},
        "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


@respx.mock
async def test_run_one_turn_returns_reply_and_threads_history(session_factory):
    respx.post(DS_URL).mock(side_effect=[_text("你好呀"), _text("第二轮回复")])
    strat = NoMemoryStrategy()
    reply1, history1 = await run_one_turn(
        persona="你是助理", user_input="你好", history=(), strategy=strat,
        employee="assistant", session=None, model="deepseek-chat", max_steps=4)
    assert reply1 == "你好呀"
    # history threads (user + assistant accumulated, minus system)
    reply2, history2 = await run_one_turn(
        persona="你是助理", user_input="再问", history=history1, strategy=strat,
        employee="assistant", session=None, model="deepseek-chat", max_steps=4)
    assert reply2 == "第二轮回复"
    assert any(m["content"] == "你好" for m in history2)


@respx.mock
async def test_run_one_turn_calls_strategy_hooks(session_factory, monkeypatch):
    respx.post(DS_URL).mock(return_value=_text("ok"))
    calls = {"prefix": 0, "after": 0}

    class SpyStrategy(NoMemoryStrategy):
        async def system_prefix(self, employee, user_msg):
            calls["prefix"] += 1
            return "记忆:你叫小明"
        async def after_turn(self, employee, session, msgs):
            calls["after"] += 1

    reply, _ = await run_one_turn(
        persona="助理", user_input="hi", history=(), strategy=SpyStrategy(),
        employee="assistant", session=None, model="deepseek-chat", max_steps=4)
    assert calls["prefix"] == 1 and calls["after"] == 1
```

- [ ] **Step 2: 跑失败 → 实现 chat.py**

`run_one_turn(*, persona, user_input, history, strategy, employee, session, model, max_steps) -> tuple[str, tuple[dict,...]]`:
1. `prefix = await strategy.system_prefix(employee, user_input)`
2. `system = persona + ("\n\n" + prefix if prefix else "")`
3. `user_msg = {"role":"user","content":user_input}`
4. `msgs = ({"role":"system","content":system},) + tuple(history) + (user_msg,)`
5. `state = AgentState.from_messages(msgs, workdir=<tempdir>, max_steps=max_steps)`(用 tempfile.mkdtemp 或传入;为简单用 `tempfile.TemporaryDirectory` 包住 run)
6. `state = await run(state, model, strategy.build_registry(ctx=...), ToolContext(workdir=...))`
7. `reply =` 末条 assistant 的 content(找 `state.messages` 里最后 role==assistant 且 content 非空)
8. `assistant_msg = {"role":"assistant","content":reply}`
9. `await strategy.after_turn(employee, session, [user_msg, assistant_msg])`
10. `new_history = tuple(history) + (user_msg, assistant_msg)`(**不含 system**——system 每轮重建)
11. 若 `session` 给了 SessionStore+sid 则 `save`(测试传 None 跳过)
12. `return reply, new_history`

> ctx 传给 build_registry:mem0/NoMemory 返回空 registry 不用 ctx,传 None 即可;Letta(M2b)才需要真 ctx。

`chat_repl(*, persona, strategy, employee, model, session_store=None, max_steps=8)`:建/取 session → while 读 input()(`exit`/EOF 退出)→ run_one_turn → print → 累积 history。

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/chat.py packages/ai-employee/tests/test_chat.py
git commit -m "feat(ai-employee): chat run_one_turn + REPL (per-turn system rebuild, history threading)"
```

---

## Task 8: CLI `chat` 子命令

**Files:** Modify `cli.py`;Test `tests/test_ai_employee_cli.py`(追加)

- [ ] **Step 1: 失败测试**(测构建 strategy 的纯函数 `make_strategy(name, sf, model)`)

```python
async def test_make_strategy_mem0_and_none(session_factory):
    from anvil_ai_employee.cli import make_strategy
    from anvil_ai_employee.memory.mem0 import Mem0Strategy
    from anvil_ai_employee.memory.strategy import NoMemoryStrategy
    assert isinstance(make_strategy("none", session_factory, "deepseek-chat"), NoMemoryStrategy)
    assert isinstance(make_strategy("mem0", session_factory, "deepseek-chat"), Mem0Strategy)
```

- [ ] **Step 2: 跑失败 → 实现**

`cli.py` 加 `make_strategy(name, sf, model)`(none→NoMemoryStrategy;mem0→Mem0Strategy(sf, embedder=FastEmbedEmbedder(), model=model))+ `chat` 子命令(`--employee assistant --memory mem0|none --model deepseek-chat --persona <str>`),`main()` 分派调 `asyncio.run(chat_repl(...))`。persona 默认给一句"你是一个有长期记忆的私人助理"。

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/cli.py packages/ai-employee/tests/test_ai_employee_cli.py
git commit -m "feat(ai-employee): CLI chat subcommand + make_strategy(none/mem0)"
```

---

## Task 9: mem0 记忆 eval(查库分层断言 + golden fixture)

**Files:** Create `eval/memory/__init__.py`、`eval/memory/golden.py`(fixture 对话)、`tests/test_memory_eval.py`

- [ ] **Step 1: 写 golden fixture**(`eval/memory/golden.py`)

一个多轮对话脚本(list of (role, content)):用户先说住北京、养猫;几轮后说搬到上海;再重复一次"我养猫"。导出 `BEIJING_TO_SHANGHAI: list[dict]`。

- [ ] **Step 2: 失败测试 test_memory_eval.py(stub Embedder + respx 录每轮 extract/reconcile,确定性,分层断言)**

覆盖 spec 必修-6 的五类断言:
```python
# 用 StubEmbedder(同 Task5)+ 录制每轮 extract/reconcile 的 structured_chat 输出,
# 跑完 golden 后:
# 1) 召回层:抽出"上海"后 vs.knn 近邻含旧"北京"行(直接调 vs.knn 断言)
# 2) 决策层:store.list_facts(employee) 里居住地恰一行含"上海"
# 3) NOOP:重复"养猫"不增加 facts 行数
# 4) 跨会话:新 employee session system_prefix 召回命中
# 5) embed 方向:spy 包 embedder,断言 ADD 调 embed_texts、recall 调 embed_query
```
(实现者按这些断言写;每轮的 structured_chat 输出按 golden 设计录好 side_effect 序列。)

- [ ] **Step 3: 跑通过**

- [ ] **Step 4: live 冒烟测试(`@pytest.mark.live`,真 deepseek + 真 FastEmbedEmbedder)** — 跑 golden 前两段(北京→上海),断言真模型抽取+reconcile 后 list_facts 居住地为上海。标 live 不进默认 CI。

- [ ] **Step 5: ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/eval packages/ai-employee/tests/test_memory_eval.py
git commit -m "test(ai-employee): mem0 memory eval — layered DB assertions on 北京→上海 golden"
```

---

## Task 10: 全仓校验 + example + CLAUDE.md + PR

**Files:** Create `examples/08-ai-employee-memory/README.md`;Modify `CLAUDE.md`

- [ ] **Step 1: example README** — 讲 mem0 哲学(编排器管记忆:召回注入 + 抽取调和)、三层记忆里 M2a 做了哪两层(session + longterm-fact)、跑法:
```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
uv run anvil-ai-employee chat --memory mem0
# 对它说"我住在北京,养了只猫" → 退出 → 再开 chat 问"我住哪" → 它记得
```
标注 M2b(Letta 对照)/ M3(skills、HITL)留待。

- [ ] **Step 2: 全仓 ruff** `uv run ruff check .` 全绿(有问题就地修+提交)。

- [ ] **Step 3: 全仓 pytest** `ANVIL_DATABASE_URL=...anvil_test ANVIL_TEST_DATABASE_URL=...anvil_test uv run pytest -m "not live" -q` 全绿;确认无跨包撞名(新测试文件基名唯一)、不破坏 code-agent(state 加法)与 kb。

- [ ] **Step 4: CLAUDE.md** 在 anvil-ai-employee 节补 M2a(mem0 记忆 + chat;新模块 memory/{vectorstore,strategy,mem0}、sessions、chat;复用校正;M2b/M3 留待)。Commit。

- [ ] **Step 5: 推分支 + PR**
```bash
git push -u origin feat/ai-employee-m2a
gh pr create --title "feat: P4-M2a AI 员工抽取式记忆(mem0 哲学)" --body "<总结 + 评审必修落实情况>"
```

---

## Self-Review

- **spec 覆盖**:resume reducer(T1)、存储扩展(T2)、store+vectorstore 假复用改新建(T3)、strategy 接口(T4)、Mem0 extract/reconcile/NOOP(T5)、session(T6)、chat 穿线(T7)、CLI(T8)、分层 eval(T9)、收尾(T10)——spec 各节有对应任务,8 条必修均落点(必修1=T1、必修2=T3-vectorstore、必修3=T3-store、必修4=T5 NOOP+window、必修5=T9 召回层断言+embed方向、必修6=T9 查库断言;必修7/8 属 M2b)。
- **占位扫描**:无 TBD;关键任务给了完整代码,T5/T9 给了断言清单与实现要点(非占位,是明确实现指令)。
- **类型一致**:`MemoryStore`/`MemoryVectorStore`/`Mem0Strategy`/`run_one_turn` 签名跨任务一致;`AgentState.resume/from_messages` T1 定义、T7 使用一致。
- **CI 雷区**:新测试文件基名(test_state_resume/test_memory_store_v2/test_strategy/test_mem0_strategy/test_sessions/test_chat/test_memory_eval)全仓唯一;T10 强制全仓 ruff+pytest 并提交修复。
