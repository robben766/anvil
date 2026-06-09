# P4-M2b Letta 自管式记忆 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development。Steps 用 `- [ ]`。

**Goal:** 给对话助理装上 Letta/MemGPT 哲学的自管式记忆——agent 自调工具管 core/recall/archival 三层 + self-paging 换页;与 M2a mem0 并排对照。

**Architecture:** 见 spec `docs/superpowers/specs/2026-06-09-ai-employee-m2b-memory-letta-design.md`。新建 `ae_core_blocks` 表 + `CoreBlockStore` + `letta_tools.py`(5 工具)+ `LettaStrategy`;chat 加 self-paging(复用 code-agent M6 compact+summarizer);CLI 加 letta。复用 M2a 的 MemoryStrategy 协议/MemoryVectorStore/SessionStore/chat 骨架。

**Tech Stack:** Python 3.12 / SQLAlchemy 异步 + pgvector / 复用 P3 harness(@tool/asyncbridge/context)、M2a memory 子包、guard、kb embedder。PG@5434。

---

## 关键既有接口(M2a 已落地,照此调用)

- `anvil_ai_employee.memory.strategy.MemoryStrategy`(协议:`build_registry(ctx)->ToolRegistry`、`async system_prefix(employee,user_msg)->str`、`async after_turn(employee,session,msgs)`)。
- `anvil_ai_employee.memory.store.MemoryStore`:`insert(*,employee,kind,content,embedding=None)->uuid`、`list_facts(*,employee,kind="fact")`。
- `anvil_ai_employee.memory.vectorstore.MemoryVectorStore.knn(*,employee,kinds,query_vec,k)->list[tuple[MemoryRow,float]]`。
- `anvil_ai_employee.sessions.SessionStore`:`create/save/load`。
- `anvil_ai_employee.chat.run_one_turn(*, persona, user_input, history, strategy, employee, session, model, max_steps)`、`chat_repl(...)`。
- `anvil_ai_employee.asyncbridge.block_on(coro)`;`anvil_code_agent.tools.base`(`@tool`/`ToolRegistry`/`ToolResult`/`ToolContext`)。
- `anvil_code_agent.harness.context`:`estimate_tokens(messages)->int`、`compact(messages, *, max_tokens, keep_recent=6, tool_cap=200, summarizer=None)->list`、`llm_summarizer(model)->callable`。
- `anvil_kb.embed.FastEmbedEmbedder`(embed_texts/embed_query,512d)。
- `anvil_ai_employee.db`:Base/MemoryRow(有 embedding)/SessionRow;conftest engine 建 ae+kb 表、autouse gateway fixture(有真 key 不塞 dummy)。
- respx mock URL = `https://api.deepseek.com/v1/chat/completions`(完整响应体含 id/model/choices[].message/finish_reason/usage;tool_calls 见 M2a test_worker/test_mem0 写法)。

---

## File Structure

- Modify `packages/ai-employee/src/anvil_ai_employee/db.py` — 新增 CoreBlockRow
- Create `packages/ai-employee/src/anvil_ai_employee/memory/coreblocks.py` — CoreBlockStore
- Create `packages/ai-employee/src/anvil_ai_employee/memory/letta_tools.py` — LettaToolContext + 5 @tool + build
- Create `packages/ai-employee/src/anvil_ai_employee/memory/letta.py` — LettaStrategy
- Modify `packages/ai-employee/src/anvil_ai_employee/chat.py` — self-paging 接线(可选 paging 参)
- Modify `packages/ai-employee/src/anvil_ai_employee/cli.py` — make_strategy 加 letta
- Create `packages/ai-employee/src/anvil_ai_employee/eval/memory/letta_golden.py` — Letta 工具序列 fixture
- tests/*

---

## Task 1: ae_core_blocks 表 + CoreBlockStore

**Files:** Modify `db.py`;Create `memory/coreblocks.py`;Test `tests/test_coreblocks.py`

- [ ] **Step 1: 失败测试 test_coreblocks.py**

```python
import pytest
from anvil_ai_employee.memory.coreblocks import CoreBlockStore

pytestmark = pytest.mark.asyncio


async def test_default_blocks_and_append(session_factory):
    store = CoreBlockStore(session_factory, char_limit=50)
    blocks = await store.get_all(employee="u1")  # auto-creates persona+human defaults
    assert set(blocks.keys()) == {"persona", "human"}
    ok = await store.append(employee="u1", label="human", text="住在北京")
    assert ok is True
    blocks = await store.get_all(employee="u1")
    assert "住在北京" in blocks["human"]


async def test_replace_substring(session_factory):
    store = CoreBlockStore(session_factory, char_limit=200)
    await store.append(employee="u1", label="human", text="住在北京")
    ok = await store.replace(employee="u1", label="human", old="北京", new="上海")
    assert ok is True
    blocks = await store.get_all(employee="u1")
    assert "上海" in blocks["human"] and "北京" not in blocks["human"]


async def test_replace_missing_old_returns_false(session_factory):
    store = CoreBlockStore(session_factory, char_limit=200)
    await store.get_all(employee="u1")
    assert await store.replace(employee="u1", label="human", old="不存在", new="x") is False


async def test_append_over_limit_returns_false(session_factory):
    store = CoreBlockStore(session_factory, char_limit=10)
    await store.get_all(employee="u1")
    assert await store.append(employee="u1", label="human", text="x" * 50) is False
```

- [ ] **Step 2: 跑失败 → 实现 db.py CoreBlockRow**

```python
class CoreBlockRow(Base):
    __tablename__ = "ae_core_blocks"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    char_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=500)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint("employee", "label", name="uq_core_employee_label"),)
```
(import UniqueConstraint from sqlalchemy;Integer 已在或加。)

- [ ] **Step 3: 实现 coreblocks.py**

```python
"""Core memory blocks (MemGPT): small, always-in-context, agent-editable text blocks
keyed by (employee, label). Default labels: persona, human."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import CoreBlockRow

DEFAULT_LABELS = ("persona", "human")


class CoreBlockStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], *, char_limit: int = 500):
        self._sf = session_factory
        self._char_limit = char_limit

    async def get_all(self, *, employee: str) -> dict[str, str]:
        """Return {label: content}; lazily create empty persona/human blocks first time."""
        async with self._sf() as s:
            async with s.begin():
                rows = (await s.execute(
                    select(CoreBlockRow).where(CoreBlockRow.employee == employee))).scalars().all()
                have = {r.label for r in rows}
                for label in DEFAULT_LABELS:
                    if label not in have:
                        s.add(CoreBlockRow(employee=employee, label=label,
                                           content="", char_limit=self._char_limit))
                rows = (await s.execute(
                    select(CoreBlockRow).where(CoreBlockRow.employee == employee))).scalars().all()
                return {r.label: r.content for r in rows}

    async def append(self, *, employee: str, label: str, text: str) -> bool:
        async with self._sf() as s:
            async with s.begin():
                row = (await s.execute(select(CoreBlockRow)
                    .where(CoreBlockRow.employee == employee)
                    .where(CoreBlockRow.label == label))).scalar_one_or_none()
                if row is None:
                    row = CoreBlockRow(employee=employee, label=label, content="",
                                       char_limit=self._char_limit)
                    s.add(row)
                    await s.flush()
                new_content = (row.content + ("\n" if row.content else "") + text)
                if len(new_content) > row.char_limit:
                    return False
                row.content = new_content
                return True

    async def replace(self, *, employee: str, label: str, old: str, new: str) -> bool:
        async with self._sf() as s:
            async with s.begin():
                row = (await s.execute(select(CoreBlockRow)
                    .where(CoreBlockRow.employee == employee)
                    .where(CoreBlockRow.label == label))).scalar_one_or_none()
                if row is None or old not in row.content:
                    return False
                candidate = row.content.replace(old, new)
                if len(candidate) > row.char_limit:
                    return False
                row.content = candidate
                return True
```

- [ ] **Step 4: 跑通过(需 PG)** `ANVIL_DATABASE_URL=...anvil_test uv run pytest packages/ai-employee/tests/test_coreblocks.py -q`

- [ ] **Step 5: ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/db.py packages/ai-employee/src/anvil_ai_employee/memory/coreblocks.py packages/ai-employee/tests/test_coreblocks.py
git commit -m "feat(ai-employee): ae_core_blocks + CoreBlockStore (MemGPT core memory)"
```

---

## Task 2: Letta 工具(5 个 @tool)

**Files:** Create `memory/letta_tools.py`;Test `tests/test_letta_tools.py`

工具用 P3 `@tool` 同步协议,内部异步走 `block_on`,闭包捕获 `LettaToolContext`。

- [ ] **Step 1: 失败测试 test_letta_tools.py**

```python
import pytest
from anvil_code_agent.tools.base import ToolContext
from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.letta_tools import LettaToolContext, build_letta_tools
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.sessions import SessionStore

pytestmark = pytest.mark.asyncio
TC = ToolContext(workdir="/tmp")


class StubEmbedder:
    def embed_texts(self, texts): return [[0.5] * 512 for _ in texts]
    def embed_query(self, text): return [0.5] * 512


def _tools(ctx):
    return {t.name: t for t in build_letta_tools(ctx)}


async def test_core_memory_replace_writes_db(session_factory):
    cb = CoreBlockStore(session_factory)
    await cb.append(employee="u1", label="human", text="住在北京")
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=None)
    res = _tools(ctx)["core_memory_replace"](
        {"label": "human", "old": "北京", "new": "上海"}, TC)
    assert res.ok
    blocks = await cb.get_all(employee="u1")
    assert "上海" in blocks["human"]


async def test_archival_insert_then_search(session_factory):
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=None)
    tools = _tools(ctx)
    assert tools["archival_insert"]({"text": "用户的生日是 5 月"}, TC).ok
    res = tools["archival_search"]({"query": "生日"}, TC)
    assert res.ok and "5 月" in res.content
    # archival lands in ae_memories kind=archival
    facts = await MemoryStore(session_factory).list_facts(employee="u1", kind="archival")
    assert len(facts) == 1


async def test_conversation_search_over_session(session_factory):
    ss = SessionStore(session_factory)
    sid = await ss.create(employee="u1")
    await ss.save(sid, [{"role": "user", "content": "我最喜欢的颜色是蓝色"},
                        {"role": "assistant", "content": "好的"}], status="active")
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=sid)
    res = _tools(ctx)["conversation_search"]({"query": "颜色"}, TC)
    assert res.ok and "蓝色" in res.content


async def test_core_append_over_limit_is_feedback_not_crash(session_factory):
    cb = CoreBlockStore(session_factory, char_limit=5)
    await cb.get_all(employee="u1")
    ctx = LettaToolContext(session_factory=session_factory, embedder=StubEmbedder(),
                           employee="u1", session_id=None, char_limit=5)
    res = _tools(ctx)["core_memory_append"]({"label": "human", "text": "x" * 50}, TC)
    assert res.ok is False and "limit" in res.content.lower() or not res.ok
```

- [ ] **Step 2: 跑失败 → 实现 letta_tools.py**

```python
"""Letta/MemGPT self-managed memory tools. The agent calls these itself (contrast mem0,
where the orchestrator manages memory). Sync @tool protocol; async DB/vec via block_on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anvil_code_agent.tools.base import Tool, ToolContext, ToolResult, tool

from anvil_ai_employee.asyncbridge import block_on
from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.store import MemoryStore
from anvil_ai_employee.memory.vectorstore import MemoryVectorStore
from anvil_ai_employee.sessions import SessionStore


@dataclass
class LettaToolContext:
    session_factory: Any
    embedder: Any
    employee: str
    session_id: Any  # uuid | None
    char_limit: int = 500
    archival_k: int = 5


def build_letta_tools(ctx: LettaToolContext) -> list[Tool]:
    cb = CoreBlockStore(ctx.session_factory, char_limit=ctx.char_limit)
    store = MemoryStore(ctx.session_factory)
    vs = MemoryVectorStore(ctx.session_factory)
    ss = SessionStore(ctx.session_factory)

    @tool(name="core_memory_append",
          description="往 core memory 块(label=persona/human)追加一行文本。超长度上限会失败。",
          params={"label": {"type": "string"}, "text": {"type": "string"}},
          required=["label", "text"])
    def core_memory_append(args, tc: ToolContext) -> ToolResult:
        ok = block_on(cb.append(employee=ctx.employee, label=args["label"], text=args["text"]))
        if not ok:
            return ToolResult(content="append 失败:超出 core 块长度上限,请改存 archival_insert。", ok=False)
        return ToolResult(content="core 块已追加。", ok=True)

    @tool(name="core_memory_replace",
          description="在 core memory 块内把子串 old 替换成 new。old 不存在会失败。",
          params={"label": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
          required=["label", "old", "new"])
    def core_memory_replace(args, tc: ToolContext) -> ToolResult:
        ok = block_on(cb.replace(employee=ctx.employee, label=args["label"],
                                 old=args["old"], new=args["new"]))
        if not ok:
            return ToolResult(content="replace 失败:old 不在块内或超长。", ok=False)
        return ToolResult(content="core 块已更新。", ok=True)

    @tool(name="archival_insert",
          description="把一段文本存入 archival 长期记忆(向量化,可日后检索)。",
          params={"text": {"type": "string"}}, required=["text"])
    def archival_insert(args, tc: ToolContext) -> ToolResult:
        emb = ctx.embedder.embed_texts([args["text"]])[0]
        block_on(store.insert(employee=ctx.employee, kind="archival",
                              content=args["text"], embedding=emb))
        return ToolResult(content="已存入 archival。", ok=True)

    @tool(name="archival_search",
          description="按语义检索 archival 长期记忆,返回最相关的若干条。",
          params={"query": {"type": "string"}}, required=["query"])
    def archival_search(args, tc: ToolContext) -> ToolResult:
        qv = ctx.embedder.embed_query(args["query"])
        hits = block_on(vs.knn(employee=ctx.employee, kinds=["archival"],
                               query_vec=qv, k=ctx.archival_k))
        if not hits:
            return ToolResult(content="archival 无匹配。", ok=True)
        return ToolResult(content="\n".join(f"- {row.content}" for row, _ in hits), ok=True)

    @tool(name="conversation_search",
          description="在本次会话的历史消息里按关键词检索过去说过的话(recall 记忆)。",
          params={"query": {"type": "string"}}, required=["query"])
    def conversation_search(args, tc: ToolContext) -> ToolResult:
        if ctx.session_id is None:
            return ToolResult(content="无会话历史可检索。", ok=True)
        msgs = block_on(ss.load(ctx.session_id))
        q = args["query"]
        hits = [m for m in msgs if isinstance(m.get("content"), str) and q in m["content"]]
        if not hits:
            return ToolResult(content="对话史无匹配。", ok=True)
        return ToolResult(content="\n".join(f"- {m['role']}: {m['content']}" for m in hits), ok=True)

    return [core_memory_append, core_memory_replace, archival_insert,
            archival_search, conversation_search]
```

- [ ] **Step 3: 跑通过(需 PG)** + ruff + Commit
```bash
git add packages/ai-employee/src/anvil_ai_employee/memory/letta_tools.py packages/ai-employee/tests/test_letta_tools.py
git commit -m "feat(ai-employee): Letta memory tools (core/archival/conversation, ACI feedback on failure)"
```

---

## Task 3: LettaStrategy

**Files:** Create `memory/letta.py`;Test `tests/test_letta_strategy.py`

- [ ] **Step 1: 失败测试**

```python
import pytest
from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.letta import LettaStrategy, LettaChatCtx

pytestmark = pytest.mark.asyncio


class StubEmbedder:
    def embed_texts(self, texts): return [[0.5] * 512 for _ in texts]
    def embed_query(self, text): return [0.5] * 512


def test_build_registry_has_five_tools(session_factory):
    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    reg = strat.build_registry(LettaChatCtx(employee="u1", session_id=None))
    names = {s["function"]["name"] for s in reg.schemas()}
    assert names == {"core_memory_append", "core_memory_replace", "archival_insert",
                     "archival_search", "conversation_search"}


async def test_system_prefix_injects_core_blocks(session_factory):
    cb = CoreBlockStore(session_factory)
    await cb.append(employee="u1", label="human", text="叫小明")
    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    prefix = await strat.system_prefix("u1", "hi")
    assert "core_memory" in prefix and "小明" in prefix


async def test_after_turn_is_noop(session_factory):
    strat = LettaStrategy(session_factory, embedder=StubEmbedder(), model="deepseek-chat")
    await strat.after_turn("u1", None, [])  # must not raise
```

- [ ] **Step 2: 跑失败 → 实现 letta.py**

```python
"""LettaStrategy — MemGPT self-managed memory: the agent edits memory via tools.
Contrast Mem0Strategy (orchestrator-managed). after_turn is a no-op: the agent already
made its edits inside the turn. Core blocks are injected each turn (read-your-write is
eventual: a core_memory_replace this turn is visible next turn's system block)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.memory.coreblocks import CoreBlockStore
from anvil_ai_employee.memory.letta_tools import LettaToolContext, build_letta_tools


@dataclass
class LettaChatCtx:
    employee: str
    session_id: Any


class LettaStrategy:
    def __init__(self, session_factory, *, embedder, model: str, char_limit: int = 500):
        self._sf = session_factory
        self._embedder = embedder
        self._model = model
        self._char_limit = char_limit
        self._cb = CoreBlockStore(session_factory, char_limit=char_limit)

    def build_registry(self, ctx: Any) -> ToolRegistry:
        employee = getattr(ctx, "employee", "assistant")
        session_id = getattr(ctx, "session_id", None)
        tctx = LettaToolContext(session_factory=self._sf, embedder=self._embedder,
                                employee=employee, session_id=session_id,
                                char_limit=self._char_limit)
        return ToolRegistry(build_letta_tools(tctx))

    async def system_prefix(self, employee: str, user_msg: str) -> str:
        blocks = await self._cb.get_all(employee=employee)
        lines = [f"[{label}] {content}" for label, content in blocks.items()]
        return (
            "<core_memory>\n" + "\n".join(lines) + "\n</core_memory>\n"
            "你有记忆工具:core_memory_append/replace 编辑常驻记忆,"
            "archival_insert/archival_search 存取长期知识,conversation_search 翻查历史。"
            "遇到关于用户的重要信息,主动用这些工具记下来。"
        )

    async def after_turn(self, employee: str, session: Any, msgs: list[dict]) -> None:
        return None  # agent self-manages memory inside the turn
```

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/memory/letta.py packages/ai-employee/tests/test_letta_strategy.py
git commit -m "feat(ai-employee): LettaStrategy — core-block injection + self-managed tool registry"
```

---

## Task 4: chat self-paging + ctx 透传

**Files:** Modify `chat.py`;Test `tests/test_chat_paging.py`

run_one_turn 现在用 ctx=None 调 build_registry;Letta 需要 employee+session_id。给 run_one_turn 加 ctx 构造 + self-paging。

- [ ] **Step 1: 失败测试 test_chat_paging.py**

```python
import pytest
from anvil_code_agent.harness.context import estimate_tokens
from anvil_ai_employee.chat import apply_self_paging

pytestmark = pytest.mark.asyncio


def test_apply_self_paging_truncates_when_over_budget():
    # build an over-budget history of tool messages
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(40):
        msgs.append({"role": "assistant", "content": "x" * 400})
        msgs.append({"role": "user", "content": "y" * 400})
    before = estimate_tokens(msgs)
    out, warned = apply_self_paging(msgs, budget=before // 2, warn_ratio=0.7, summarizer=None)
    assert estimate_tokens(out) <= before  # shrunk or equal
    assert out[0]["role"] == "system"  # system protected


def test_apply_self_paging_warns_near_budget():
    msgs = [{"role": "system", "content": "s" * 100}, {"role": "user", "content": "u" * 100}]
    # tokens ~ 50; budget 60, 0.7*60=42 → over warn but under flush
    out, warned = apply_self_paging(msgs, budget=60, warn_ratio=0.7, summarizer=None)
    assert warned is True
    assert any("上下文" in (m.get("content") or "") for m in out if m["role"] == "system")
```

- [ ] **Step 2: 跑失败 → 实现 chat.py**

加 `apply_self_paging(msgs, *, budget, warn_ratio=0.7, summarizer=None) -> tuple[list[dict], bool]`:
```python
from anvil_code_agent.harness.context import compact, estimate_tokens

def apply_self_paging(msgs, *, budget, warn_ratio=0.7, summarizer=None):
    tokens = estimate_tokens(msgs)
    warned = False
    out = list(msgs)
    if tokens >= budget:                      # flush: recursive summary (M6 compact)
        out = compact(out, max_tokens=budget, summarizer=summarizer)
    elif tokens >= int(budget * warn_ratio):  # warning: nudge agent to persist
        warned = True
        out = [out[0], {"role": "system",
                "content": "[系统提示] 上下文将满,请用 archival_insert / core_memory_replace 保存要点。"}
               ] + out[1:]
    return out, warned
```
并在 `run_one_turn` 里:加可选参 `paging: dict | None = None`(含 budget/warn_ratio/summarizer);构造 msgs 后,若 paging 给了则 `msgs, _ = apply_self_paging(list(msgs), **paging)`。并把 build_registry 的 ctx 从 None 改成携带 employee+session_id 的对象(用一个轻量 `SimpleNamespace(employee=employee, session_id=<from session>)` 或新 dataclass);mem0/None 的 build_registry 忽略 ctx 不受影响。

> 注意:run_one_turn 已有签名,加 `paging=None` 与 ctx 构造是**加法**,保 M2a 的 test_chat.py 不破(默认 paging=None=不分页,ctx 传了但 mem0/None 忽略)。

- [ ] **Step 3: 跑通过(test_chat_paging + M2a test_chat 都要绿)** + ruff + Commit
```bash
git add packages/ai-employee/src/anvil_ai_employee/chat.py packages/ai-employee/tests/test_chat_paging.py
git commit -m "feat(ai-employee): chat self-paging (warn + flush via M6 compact) + memory ctx threading"
```

---

## Task 5: CLI letta 选项

**Files:** Modify `cli.py`;Test `tests/test_ai_employee_cli.py`(追加)

- [ ] **Step 1: 追加失败测试**

```python
async def test_make_strategy_letta(session_factory):
    from anvil_ai_employee.cli import make_strategy
    from anvil_ai_employee.memory.letta import LettaStrategy
    assert isinstance(make_strategy("letta", session_factory, "deepseek-chat"), LettaStrategy)
```

- [ ] **Step 2: 跑失败 → 实现**

`make_strategy` 加 `"letta"` 分支 → `LettaStrategy(sf, embedder=FastEmbedEmbedder(), model=model)`;`chat` 子命令 `--memory` choices 加 `letta`。Letta 路径在 chat_repl 里开启 self-paging(传 paging dict,summarizer=llm_summarizer(model));mem0/none 不传(保 M2a 行为)。**为简单**:chat_repl 可统一根据 strategy 类型决定是否传 paging——或加一个 `paging` 参由 cli 决定。实现者择简:cli 对 letta 构造 paging dict 传给 chat_repl。

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/cli.py packages/ai-employee/tests/test_ai_employee_cli.py
git commit -m "feat(ai-employee): CLI --memory letta + self-paging wiring"
```

---

## Task 6: Letta eval(agent 自调工具,不假设同轮可见)

**Files:** Create `eval/memory/letta_golden.py`;Test `tests/test_letta_eval.py`

- [ ] **Step 1: 写 letta_golden.py** — 导出一段"agent 自己调工具"的剧本说明(注释 + 期望工具序列),供测试构造 side_effect。

- [ ] **Step 2: 失败测试 test_letta_eval.py**

用 respx 录一段 agent 工具调用序列(assistant 调 core_memory_replace 改 human → 调 archival_insert 存事实 → 文本收尾),驱动 `run_one_turn(strategy=LettaStrategy)`,断言:
```python
# 1) DB 变更:CoreBlockStore.get_all human 块被改;MemoryStore.list_facts(kind="archival") 有新行
# 2) 不假设同轮可见:只在 run_one_turn 返回后查库,不断言回复文本里反映了改动
# 3) archival_search 召回:archival_insert 后用 LettaToolContext 直接调 archival_search 命中
# 4) self-paging:构造超 budget 历史 → apply_self_paging(summarizer=stub) → token 降 + 无孤儿 tool 消息
```
(stub summarizer 返回固定串;StubEmbedder 确定性。)

- [ ] **Step 3: 跑通过**

- [ ] **Step 4: live 冒烟(`@pytest.mark.live`)** — 真 deepseek 驱动 LettaStrategy:对它说一件关于用户的事,断言它**自己**调了某记忆工具(core/archival 之一),DB 有变更;下一轮能召回。标 live 不进 CI。

- [ ] **Step 5: ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/eval/memory/letta_golden.py packages/ai-employee/tests/test_letta_eval.py
git commit -m "test(ai-employee): Letta eval — agent self-calls memory tools, eventual read-your-write"
```

---

## Task 7: 全仓校验 + example + CLAUDE.md + PR

**Files:** Create `examples/09-ai-employee-letta-memory/README.md`;Modify `CLAUDE.md`

- [ ] **Step 1: example README** — 讲 Letta 哲学(agent 自管记忆:core/recall/archival 三层 + 5 工具 + self-paging),与 M2a mem0 的对照表,跑法:
```bash
uv run anvil-ai-employee chat --memory letta
# 对它说"我叫小明,住在上海" → 它自己调 core_memory_replace 写进 human 块
# 退出再开 → 它的 system 里带着 core 块,记得你
```
标注 mem0(M2a)vs letta(M2b)对照,M3+ 留待。

- [ ] **Step 2: 全仓 ruff** `uv run ruff check .` 全绿。

- [ ] **Step 3: 全仓 pytest** `ANVIL_DATABASE_URL=...anvil_test ANVIL_TEST_DATABASE_URL=...anvil_test uv run pytest -m "not live" -q` 全绿;新测试文件基名唯一(test_coreblocks/test_letta_tools/test_letta_strategy/test_chat_paging/test_letta_eval),不破坏 M2a/code-agent/kb。

- [ ] **Step 4: CLAUDE.md** anvil-ai-employee 节加 M2b(Letta 自管式:5 工具 + core blocks + self-paging + 三层映射;CLI --memory letta;M3+ 留待)。Commit。

- [ ] **Step 5: 推 + PR**
```bash
git push -u origin feat/ai-employee-m2b
gh pr create --title "feat: P4-M2b AI 员工自管式记忆(Letta 哲学)" --body "<总结 + mem0 对照 + 评审必修落实>"
```

---

## Self-Review

- **spec 覆盖**:core blocks(T1)、5 工具(T2)、LettaStrategy(T3)、self-paging+ctx(T4)、CLI(T5)、eval(T6)、收尾(T7)——spec 各节有对应任务。评审 Letta 必修:recall 层=T2 conversation_search;self-paging=T4;读己写最终一致=T3 docstring + T6 eval 不假设同轮可见;heartbeat 映射=T3 docstring。
- **占位扫描**:无 TBD;关键任务给完整代码,T6 给断言清单。
- **类型一致**:`LettaToolContext`/`LettaChatCtx`/`CoreBlockStore`/`build_letta_tools` 跨任务一致;复用 M2a 的 MemoryStore/VectorStore/SessionStore/MemoryStrategy 签名不变。
- **CI 雷区**:新测试基名全仓唯一;chat.py/run_one_turn 改动是加法(paging=None 默认),M2a test_chat 不破;db.py 加表不影响既有。T7 全仓 ruff+pytest 强制。
