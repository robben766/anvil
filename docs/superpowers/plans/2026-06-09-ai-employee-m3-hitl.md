# P4-M3 Agent Inbox(HITL)+ skills 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development。Steps 用 `- [ ]`。

**Goal:** 高风险动作挂起 → Agent Inbox 等人 approve/edit/reject/respond → 恢复继续;每次干预写回长期记忆。顺带 skills-as-markdown。

**Architecture:** 见 spec `docs/superpowers/specs/2026-06-09-ai-employee-m3-hitl-inbox-design.md`。核心:`hitl.py`(hitl_step/hitl_run/apply_decision,挂起=最后 assistant 消息里未答的 tool_call,不改 P3 step)+ `ae_inbox` 表 + `InboxStore` + CLI inbox + skills 加载。复用 P3 permission/recovery/tools.base/gateway + M1 queue/worker + M2a MemoryStore。

**Tech Stack:** Python 3.12 / SQLAlchemy 异步 / 复用 P3 harness、M1/M2a。PG@5434。

---

## 关键既有接口

- `anvil_code_agent.state.AgentState`(frozen:messages tuple/step/max_steps/workdir/status;new/append/advance/finish;M2a 加的 resume/from_messages)。状态字符串 status 任意,可用 "suspended"。
- `anvil_code_agent.harness.recovery.dump_state(state)->dict` / `load_state(dict)->AgentState`。
- `anvil_code_agent.harness.permission.risk_level(name)->str`(unknown→high)。
- `anvil_code_agent.tools.base.ToolRegistry.dispatch(name,args,ctx)->ToolResult`;`ToolContext`;`@tool`。
- `anvil_gateway.chat(model, messages, *, tools=...)`(返回有 .raw / .tool_calls;tool_calls 结构见 loop.py:31-44)。
- `anvil_obs.span`。
- M1:`anvil_ai_employee.db`(Base/JobRow/...)、`scheduler.queue`、`worker`、`cli`。
- M2a:`anvil_ai_employee.memory.store.MemoryStore.insert(*,employee,kind,content,embedding=None)`;`anvil_kb.embed.FastEmbedEmbedder`。
- conftest:engine 建 ae+kb 表、autouse gateway fixture(真 key 不塞 dummy);respx URL `https://api.deepseek.com/v1/chat/completions`。

---

## File Structure

- Modify `db.py` — 新增 InboxRow
- Create `memory/`(无)… 实为顶层 `hitl.py` — HitlDecision/policy/_unanswered_tool_calls/hitl_step/hitl_run/apply_decision
- Create `inbox.py` — InboxStore
- Create `skills_loader.py` — load_skill + skills/*.md
- Modify `cli.py` — inbox 子命令 + run-hitl demo
- tests/*

---

## Task 1: hitl_step / hitl_run / _unanswered_tool_calls(纯 harness,respx mock)

**Files:** Create `hitl.py`;Test `tests/test_hitl_loop.py`

- [ ] **Step 1: 失败测试 test_hitl_loop.py**

```python
import json
import httpx
import pytest
import respx
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool
from anvil_ai_employee.hitl import HitlDecision, hitl_run, suspend_high, _unanswered_tool_calls

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"
TC = ToolContext(workdir="/tmp")


def _tool_call(tcid, name, args):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": tcid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
         "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _text(t):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": t}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _registry():
    @tool(name="safe_echo", description="d", params={"t": {"type": "string"}}, required=["t"])
    def safe_echo(args, ctx):
        return ToolResult(content="echo:" + args["t"], ok=True)

    @tool(name="danger", description="d", params={"t": {"type": "string"}}, required=["t"])
    def danger(args, ctx):
        return ToolResult(content="DID:" + args["t"], ok=True)
    return ToolRegistry([safe_echo, danger])


def _policy(name, args, risk):
    # danger is unknown→high→suspend; safe_echo unknown→high too — force safe low here
    return HitlDecision.SUSPEND if name == "danger" else HitlDecision.EXECUTE


@respx.mock
async def test_runs_safe_tool_to_done():
    respx.post(DS_URL).mock(side_effect=[_tool_call("c1", "safe_echo", {"t": "hi"}), _text("done")])
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=10)
    out = await hitl_run(s, "deepseek-chat", _registry(), TC, policy=_policy)
    assert out.status == "done"
    assert any(m.get("content") == "echo:hi" for m in out.messages if m["role"] == "tool")


@respx.mock
async def test_suspends_on_high_risk():
    respx.post(DS_URL).mock(side_effect=[_tool_call("c1", "danger", {"t": "rm"})])
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=10)
    out = await hitl_run(s, "deepseek-chat", _registry(), TC, policy=_policy)
    assert out.status == "suspended"
    pending = _unanswered_tool_calls(out)
    assert len(pending) == 1 and pending[0]["function"]["name"] == "danger"
    # the danger tool was NOT executed (no tool message for it)
    assert not any(m["role"] == "tool" for m in out.messages)
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现 hitl.py**

```python
"""HITL agent loop: suspend on high-risk tool calls for human review, resume after.
The pending action is simply the unanswered tool_call in the last assistant message —
AgentState (via recovery.dump_state) fully captures the suspension point. We do NOT fork
P3's step(); hitl_step does one thing per call (process one pending tool, or one model
call) so suspension is a clean return."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from typing import Any

from anvil_code_agent.harness.permission import risk_level
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry
from anvil_gateway import chat
from anvil_obs import span


class HitlDecision(str, Enum):
    EXECUTE = "execute"
    SUSPEND = "suspend"
    DENY = "deny"


HitlPolicy = Callable[[str, dict[str, Any], str], HitlDecision]


def suspend_high(name: str, args: dict[str, Any], risk: str) -> HitlDecision:
    return HitlDecision.SUSPEND if risk == "high" else HitlDecision.EXECUTE


def _last_assistant_with_calls(messages) -> dict | None:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return m
        if m.get("role") == "tool":
            continue
        if m.get("role") == "assistant":
            return None  # a plain assistant message after = no pending
    return None


def _unanswered_tool_calls(state: AgentState) -> list[dict]:
    msg = _last_assistant_with_calls(state.messages)
    if msg is None:
        return []
    answered = {m.get("tool_call_id") for m in state.messages if m.get("role") == "tool"}
    return [tc for tc in msg["tool_calls"] if tc["id"] not in answered]


def _args_of(tc: dict) -> dict[str, Any]:
    try:
        return json.loads(tc["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}


def _tool_msg(tcid: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tcid, "content": content}


async def hitl_step(state, model, registry, ctx, *, policy: HitlPolicy) -> AgentState:
    pending = _unanswered_tool_calls(state)
    if pending:
        tc = pending[0]
        name = tc["function"]["name"]
        args = _args_of(tc)
        risk = risk_level(name)
        d = policy(name, args, risk)
        if d == HitlDecision.SUSPEND:
            return state.finish("suspended")
        if d == HitlDecision.DENY:
            return state.append(_tool_msg(tc["id"], f"denied by policy (risk={risk})"))
        with span("ai_employee.hitl.tool", tool=name, risk=risk):
            result = registry.dispatch(name, args, ctx)
        return state.append(_tool_msg(tc["id"], result.content))
    resp = await chat(model, list(state.messages), tools=registry.schemas())
    assistant = resp.raw["choices"][0]["message"]
    if resp.tool_calls:
        return state.append(assistant).advance()
    return state.append(assistant).advance().finish("done")


async def hitl_run(state, model, registry, ctx, *, policy: HitlPolicy = suspend_high) -> AgentState:
    with span("ai_employee.hitl.run", model=model):
        while state.status == "running":
            if state.step >= state.max_steps:
                return state.finish("exhausted")
            state = await hitl_step(state, model, registry, ctx, policy=policy)
            if state.status == "suspended":
                return state
        return state
```

> 注:`advance()` 只在调模型那次发生(对齐 max_steps);处理待答工具的分支不 advance(否则一个高风险动作连累 step 计数)。`_last_assistant_with_calls` 找最后一条带 tool_calls 的 assistant,但要确保它后面没有"新的纯 assistant 文本"(那种情况无 pending)。

- [ ] **Step 4: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/hitl.py packages/ai-employee/tests/test_hitl_loop.py
git commit -m "feat(ai-employee): hitl_step/hitl_run — suspend on high-risk, pending=unanswered tool_call"
```

---

## Task 2: apply_decision(恢复:把人的决策变成一条 tool 消息)

**Files:** Modify `hitl.py`;Test `tests/test_hitl_resume.py`

- [ ] **Step 1: 失败测试 test_hitl_resume.py**

```python
import json
import httpx
import pytest
import respx
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool
from anvil_ai_employee.hitl import HitlDecision, apply_decision, hitl_run

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"
TC = ToolContext(workdir="/tmp")


def _registry(calls):
    @tool(name="danger", description="d", params={"t": {"type": "string"}}, required=["t"])
    def danger(args, ctx):
        calls.append(args["t"])
        return ToolResult(content="DID:" + args["t"], ok=True)
    return ToolRegistry([danger])


def _suspended_state():
    # an assistant message proposing danger(t=rm), unanswered → suspended
    msgs = (
        {"role": "system", "content": "s"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "danger", "arguments": json.dumps({"t": "rm"})}}]},
    )
    return AgentState(messages=msgs, step=1, max_steps=10, workdir="/tmp", status="suspended")


async def test_approve_executes_original_args():
    calls = []
    st = apply_decision(_suspended_state(), decision="approve", payload={}, registry=_registry(calls), ctx=TC)
    assert st.status == "running"
    assert calls == ["rm"]  # executed with original args
    assert any(m["role"] == "tool" and "DID:rm" in m["content"] for m in st.messages)


async def test_edit_executes_new_args():
    calls = []
    st = apply_decision(_suspended_state(), decision="edit", payload={"args": {"t": "safe"}},
                        registry=_registry(calls), ctx=TC)
    assert calls == ["safe"]


async def test_reject_injects_feedback_no_exec():
    calls = []
    st = apply_decision(_suspended_state(), decision="reject", payload={"reason": "太危险"},
                        registry=_registry(calls), ctx=TC)
    assert calls == []
    assert any(m["role"] == "tool" and "太危险" in m["content"] for m in st.messages)
    assert st.status == "running"


async def test_respond_injects_custom_no_exec():
    calls = []
    st = apply_decision(_suspended_state(), decision="respond", payload={"message": "我帮你做了"},
                        registry=_registry(calls), ctx=TC)
    assert calls == []
    assert any(m["role"] == "tool" and "我帮你做了" in m["content"] for m in st.messages)


@respx.mock
async def test_resume_continues_to_done():
    # after approve, model wraps up
    respx.post(DS_URL).mock(return_value=httpx.Response(200, json={"id": "x", "model": "deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "完成"},
        "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    calls = []
    reg = _registry(calls)
    st = apply_decision(_suspended_state(), decision="approve", payload={}, registry=reg, ctx=TC)
    out = await hitl_run(st, "deepseek-chat", reg, TC, policy=lambda n, a, r: HitlDecision.EXECUTE)
    assert out.status == "done"
```

- [ ] **Step 2: 跑失败 → 在 hitl.py 加 apply_decision**

```python
def apply_decision(state, *, decision: str, payload: dict, registry: ToolRegistry,
                   ctx: ToolContext) -> AgentState:
    pending = _unanswered_tool_calls(state)
    if not pending:
        raise ValueError("apply_decision: no pending tool call in state")
    tc = pending[0]
    tcid = tc["id"]
    name = tc["function"]["name"]
    if decision == "approve":
        content = registry.dispatch(name, _args_of(tc), ctx).content
    elif decision == "edit":
        content = registry.dispatch(name, payload["args"], ctx).content
    elif decision == "reject":
        content = f"[人工拒绝] {payload.get('reason', '')}"
    elif decision == "respond":
        content = payload["message"]
    else:
        raise ValueError(f"unknown decision: {decision}")
    msgs = state.messages + (_tool_msg(tcid, content),)
    return replace(state, messages=msgs, status="running")
```

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/hitl.py packages/ai-employee/tests/test_hitl_resume.py
git commit -m "feat(ai-employee): apply_decision — approve/edit/reject/respond resume into a tool message"
```

---

## Task 3: ae_inbox 表 + InboxStore

**Files:** Modify `db.py`;Create `inbox.py`;Test `tests/test_inbox.py`

- [ ] **Step 1: 失败测试 test_inbox.py**

```python
import pytest
from anvil_code_agent.state import AgentState
from anvil_ai_employee.inbox import InboxStore

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
```

- [ ] **Step 2: 跑失败 → db.py 加 InboxRow**

```python
class InboxRow(Base):
    __tablename__ = "ae_inbox"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    employee: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False)
    tool_args: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    risk: Mapped[str] = mapped_column(Text, nullable=False)
    state_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    decision: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 3: 实现 inbox.py**

```python
"""Agent Inbox store: persist a suspended agent for human review and resolution."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from anvil_code_agent.harness.permission import risk_level
from anvil_code_agent.harness.recovery import dump_state
from anvil_code_agent.state import AgentState

from anvil_ai_employee.db import InboxRow
from anvil_ai_employee.hitl import _args_of, _unanswered_tool_calls


class InboxStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def suspend(self, *, employee: str, state: AgentState, job_id=None) -> uuid.UUID:
        pending = _unanswered_tool_calls(state)
        if not pending:
            raise ValueError("suspend: state has no pending tool call")
        tc = pending[0]
        name = tc["function"]["name"]
        iid = uuid.uuid4()
        async with self._sf() as s:
            async with s.begin():
                s.add(InboxRow(id=iid, job_id=job_id, employee=employee, tool_name=name,
                               tool_args=_args_of(tc), risk=risk_level(name),
                               state_json=dump_state(state), status="pending"))
        return iid

    async def list_pending(self, *, employee=None) -> list[InboxRow]:
        async with self._sf() as s:
            q = select(InboxRow).where(InboxRow.status == "pending").order_by(InboxRow.created_at)
            if employee is not None:
                q = q.where(InboxRow.employee == employee)
            return list((await s.execute(q)).scalars().all())

    async def get(self, inbox_id) -> InboxRow | None:
        async with self._sf() as s:
            return (await s.execute(select(InboxRow).where(InboxRow.id == inbox_id))).scalar_one_or_none()

    async def resolve(self, inbox_id, *, decision: str, payload: dict) -> None:
        async with self._sf() as s:
            async with s.begin():
                await s.execute(sa_update(InboxRow).where(InboxRow.id == inbox_id)
                    .where(InboxRow.status == "pending")  # idempotent: only resolve once
                    .values(status="resolved", decision=decision, decision_payload=payload,
                            resolved_at=func.now()))
```

- [ ] **Step 4: 跑通过(需 PG)+ ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/db.py packages/ai-employee/src/anvil_ai_employee/inbox.py packages/ai-employee/tests/test_inbox.py
git commit -m "feat(ai-employee): ae_inbox + InboxStore (suspend/list/resolve, idempotent)"
```

---

## Task 4: 干预写回长期记忆

**Files:** Create `memory_log.py`(或加进 inbox.py);Test `tests/test_intervention_memory.py`

- [ ] **Step 1: 失败测试**

```python
import pytest
from anvil_ai_employee.hitl_memory import record_intervention
from anvil_ai_employee.memory.store import MemoryStore

pytestmark = pytest.mark.asyncio


class StubEmbedder:
    def embed_texts(self, texts): return [[0.5] * 512 for _ in texts]
    def embed_query(self, text): return [0.5] * 512


async def test_records_reject_as_memory(session_factory):
    await record_intervention(session_factory, embedder=StubEmbedder(), employee="assistant",
                              tool_name="bash", decision="reject",
                              payload={"reason": "危险"}, tool_args={"cmd": "rm"})
    facts = await MemoryStore(session_factory).list_facts(employee="assistant", kind="hitl")
    assert len(facts) == 1 and "拒绝" in facts[0].content and "bash" in facts[0].content
    assert facts[0].embedding is not None  # recallable
```

- [ ] **Step 2: 跑失败 → 实现 hitl_memory.py**

```python
"""Write each human intervention back to long-term memory (越用越懂你): future runs can
recall how the human decided last time. Stored kind='hitl' with embedding so mem0 recall
picks it up."""

from __future__ import annotations

from anvil_ai_employee.memory.store import MemoryStore


def _phrase(employee, tool_name, decision, payload, tool_args) -> str:
    if decision == "approve":
        return f"审批人批准了 {employee} 的 {tool_name} 操作(参数 {tool_args})。"
    if decision == "edit":
        return f"审批人把 {employee} 的 {tool_name} 参数改成 {payload.get('args')}。"
    if decision == "reject":
        return f"审批人拒绝了 {employee} 的 {tool_name} 操作,原因:{payload.get('reason', '')}。"
    if decision == "respond":
        return f"对 {employee} 的 {tool_name},审批人直接答复:{payload.get('message', '')}。"
    return f"对 {tool_name} 的未知决策 {decision}。"


async def record_intervention(session_factory, *, embedder, employee, tool_name,
                              decision, payload, tool_args) -> None:
    text = _phrase(employee, tool_name, decision, payload, tool_args)
    emb = embedder.embed_texts([text])[0]
    await MemoryStore(session_factory).insert(
        employee=employee, kind="hitl", content=text, embedding=emb)
```

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/hitl_memory.py packages/ai-employee/tests/test_intervention_memory.py
git commit -m "feat(ai-employee): record_intervention — each HITL decision becomes recallable memory"
```

---

## Task 5: resume_from_inbox 串联(resolve→恢复→再挂起/done)

**Files:** Create `inbox_resume.py`;Test `tests/test_inbox_resume.py`

- [ ] **Step 1: 失败测试**(respx mock + 真 InboxStore + StubEmbedder)

```python
# 流程:suspend(state) → InboxStore 有 pending → resolve(approve) → resume_from_inbox 跑到 done(或再挂起)
# 断言:resume 后 job/状态 done;记忆里有 hitl 干预记录;工具按 approve 执行
```
具体:构造一个挂起 state(danger 待审)→ InboxStore.suspend → resolve(decision="approve")→ `resume_from_inbox(inbox_row, registry, ctx, model, *, session_factory, embedder, policy)`:load_state → record_intervention → apply_decision → hitl_run → 返回终态。respx 让 approve 后模型收尾 done。断言终态 done、danger 执行了、MemoryStore kind="hitl" 有 1 条。

- [ ] **Step 2: 实现 inbox_resume.py**

```python
"""Resume a resolved inbox item: apply the human decision, log it to memory, continue
the HITL loop until the next suspension or completion."""

from __future__ import annotations

from anvil_code_agent.harness.recovery import load_state

from anvil_ai_employee.hitl import apply_decision, hitl_run, suspend_high
from anvil_ai_employee.hitl_memory import record_intervention


async def resume_from_inbox(inbox_row, *, registry, ctx, model, session_factory, embedder,
                            policy=suspend_high):
    state = load_state(inbox_row.state_json)
    await record_intervention(
        session_factory, embedder=embedder, employee=inbox_row.employee,
        tool_name=inbox_row.tool_name, decision=inbox_row.decision,
        payload=inbox_row.decision_payload or {}, tool_args=inbox_row.tool_args)
    state = apply_decision(state, decision=inbox_row.decision,
                           payload=inbox_row.decision_payload or {}, registry=registry, ctx=ctx)
    return await hitl_run(state, model, registry, ctx, policy=policy)
```

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/inbox_resume.py packages/ai-employee/tests/test_inbox_resume.py
git commit -m "feat(ai-employee): resume_from_inbox — log intervention + apply decision + continue"
```

---

## Task 6: CLI inbox 子命令 + run-hitl demo

**Files:** Modify `cli.py`;Test `tests/test_ai_employee_cli.py`(追加纯函数测试)

- [ ] **Step 1: 失败测试**(测可测的纯异步函数,不起交互)

```python
async def test_inbox_list_and_resolve_helpers(session_factory):
    # 用 InboxStore 直接造一个 pending,测 cli 的 inbox_list_text / inbox_resolve 纯函数
    import json
    from anvil_code_agent.state import AgentState
    from anvil_ai_employee.inbox import InboxStore
    from anvil_ai_employee.cli import inbox_list_text, inbox_resolve
    msgs = ({"role": "system", "content": "s"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": json.dumps({"cmd": "rm"})}}]})
    st = AgentState(messages=msgs, step=1, max_steps=10, workdir="/tmp", status="suspended")
    iid = await InboxStore(session_factory).suspend(employee="assistant", state=st)
    txt = await inbox_list_text(session_factory)
    assert "bash" in txt and str(iid)[:8] in txt
    await inbox_resolve(session_factory, inbox_id=iid, decision="reject", payload={"reason": "no"})
    assert "(空)" in await inbox_list_text(session_factory) or "pending" not in (await inbox_list_text(session_factory)).lower()
```

- [ ] **Step 2: 实现 cli.py**

加纯函数:`inbox_list_text(sf)->str`(列 pending)、`inbox_resolve(sf, *, inbox_id, decision, payload)`(InboxStore.resolve;**M3 简化**:resolve 即记一条干预记忆——调 record_intervention 用 FastEmbedEmbedder,需要 inbox_row 的 tool_name/tool_args)。子命令:
- `inbox list`
- `inbox approve <id>` / `inbox edit <id> --args '<json>'` / `inbox reject <id> --reason '...'` / `inbox respond <id> --message '...'`
- `run-hitl --persona ... --task ...`(demo:用一个含高风险工具的小 registry 跑 hitl_run 到挂起,写 InboxStore,打印 inbox id)

`main()` 分派。inbox_resolve 内部组 payload(reject→{reason}、edit→{args}、respond→{message}、approve→{})。

> 注意:resolve 后的"恢复执行"在 CLI 里可选(可打印"已记录决策,下次 worker 恢复")——M3 为聚焦,resolve 落库 + 写记忆即可,完整 worker 自动恢复接线标为后续(resume_from_inbox 已具备,可在 run-hitl demo 里手动串一次证明闭环)。

- [ ] **Step 3: 跑通过(CLI 测试 + 既有都绿)+ ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/cli.py packages/ai-employee/tests/test_ai_employee_cli.py
git commit -m "feat(ai-employee): CLI inbox list/approve/edit/reject/respond + run-hitl demo"
```

---

## Task 7: skills-as-markdown(M2 砍出来的)

**Files:** Create `skills_loader.py` + `skills/assistant.md`、`skills/kb_reporter.md`;Test `tests/test_skills_loader.py`

- [ ] **Step 1: 失败测试**

```python
from anvil_ai_employee.skills_loader import load_skill, available_skills


def test_load_existing_skill():
    text = load_skill("assistant")
    assert isinstance(text, str) and len(text) > 0


def test_unknown_skill_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_skill("nope")


def test_available_lists_md_files():
    skills = available_skills()
    assert "assistant" in skills
```

- [ ] **Step 2: 实现 skills_loader.py + 两个 .md**

```python
"""Skills tier: versioned markdown persona/skill files loaded at runtime (the third tier
of the three-tier memory). M1/M2 hard-coded personas in Python; these externalize them."""

from __future__ import annotations

from pathlib import Path

_SKILLS_DIR = Path(__file__).parent / "skills"


def load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"skill not found: {name} (looked in {_SKILLS_DIR})")
    return path.read_text(encoding="utf-8").strip()


def available_skills() -> list[str]:
    return sorted(p.stem for p in _SKILLS_DIR.glob("*.md"))
```
`skills/assistant.md`:对话助理 persona(可呼应 M2 的"有长期记忆的私人助理")。`skills/kb_reporter.md`:把 M1 kb_digest 的 PERSONA 文本搬过来。
确保 `skills/*.md` 随包打包:`pyproject.toml` 的 hatch wheel 默认含 src 下文件;若 .md 不被打包,加 `[tool.hatch.build.targets.wheel.force-include]` 或确认 package-data。本地测试用文件路径即可。

- [ ] **Step 3: 跑通过 + ruff + Commit**
```bash
git add packages/ai-employee/src/anvil_ai_employee/skills_loader.py packages/ai-employee/src/anvil_ai_employee/skills/ packages/ai-employee/tests/test_skills_loader.py
git commit -m "feat(ai-employee): skills-as-markdown loader (third memory tier, M2-deferred)"
```

---

## Task 8: 全仓校验 + example + CLAUDE.md + PR

- [ ] **Step 1: example** `examples/10-ai-employee-hitl/README.md` — 讲 HITL 防跑飞(挂起=未答 tool_call、四动作、干预写回记忆)、跑法(run-hitl 跑到挂起 → inbox list → approve/reject → 恢复),与 M2 记忆的衔接(干预喂记忆)。skills 一节。
- [ ] **Step 2: 全仓 ruff** `uv run ruff check .` 全绿。
- [ ] **Step 3: 全仓 pytest** `ANVIL_DATABASE_URL=...anvil_test ANVIL_TEST_DATABASE_URL=...anvil_test uv run pytest -m "not live" -q` 全绿;新测试基名唯一(test_hitl_loop/test_hitl_resume/test_inbox/test_intervention_memory/test_inbox_resume/test_skills_loader),不破坏 M1/M2/code-agent/kb。
- [ ] **Step 4: CLAUDE.md** 加 M3 节(HITL 挂起恢复 + Agent Inbox + 干预写记忆 + skills;复用 permission/recovery;M4/M5 留待)。Commit。
- [ ] **Step 5: 推 + PR**
```bash
git push -u origin feat/ai-employee-m3
gh pr create --title "feat: P4-M3 Agent Inbox(HITL 防跑飞)+ skills" --body "<总结 + 机制 + 复用>"
```

---

## Self-Review

- **spec 覆盖**:hitl_step/run(T1)、apply_decision 四动作(T2)、ae_inbox+store(T3)、干预写记忆(T4)、resume 串联(T5)、CLI inbox(T6)、skills(T7)、收尾(T8)——spec 各节有对应任务。
- **占位扫描**:无 TBD;关键任务给完整代码,T5/T6 给断言/实现要点。
- **类型一致**:`_unanswered_tool_calls`/`apply_decision`/`InboxStore`/`record_intervention`/`resume_from_inbox` 跨任务签名一致;复用 recovery.dump/load_state、MemoryStore.insert 签名不变。
- **CI 雷区**:新测试基名全仓唯一;db.py 加表/code-agent 不动(HITL 全在 ai-employee);T8 全仓 ruff+pytest 强制;skills/*.md 打包确认。
