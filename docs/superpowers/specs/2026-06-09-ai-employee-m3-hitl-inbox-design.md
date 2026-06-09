# 产品④ AI 员工 — M3「Agent Inbox(HITL 防跑飞)+ skills」设计

> 蓝图 §4.5 的"防跑飞统一范式"。高风险动作默认**挂起**等人 approve/edit/reject/respond,每次干预**写回长期记忆**(喂 M2 的 mem0/Letta,越用越懂你)。复用 P3 的 `permission.py`(风险分级)+ `recovery.py`(挂起-恢复持久化)+ M1 的 queue/worker。顺带做掉 M2 砍出来的 skills-as-markdown(persona 文件化)。

## 目标(一句话)

让员工跑到高风险工具调用时**挂起**(persist 完整状态),进 Agent Inbox 等人决策(批准/改参/拒绝/代答),人决策后**恢复**继续跑;每次人工干预写成一条长期记忆。

## 核心机制:挂起 = "最后一条 assistant 消息里尚无 tool 回复的 tool_call"

**关键洞察**:agent 状态(`recovery.dump_state` 可序列化的 messages/step/status)本身就完整承载挂起点——**待审动作 = 最后一条 assistant 消息的 tool_calls 里,还没有对应 `role=tool` 回复的那个**。不需要额外存"pending action"字段。

因此 HITL 不改 P3 的 `step()`(保持其干净),而在 ai-employee 实现一个并行的 `hitl_step`——**每步只做一件事**:处理一个待答工具,或调一次模型。遇高风险待审工具就 `finish("suspended")`。

```python
# 风险决策:三态(对照 P3 的 bool policy)
class HitlDecision(str, Enum):
    EXECUTE = "execute"   # 直接执行(low/medium 自动)
    SUSPEND = "suspend"   # 挂起等人(high)
    DENY    = "deny"      # 直接拒(可选策略)

HitlPolicy = Callable[[str, dict, str], HitlDecision]
# 默认:suspend_high = low/medium→EXECUTE, high→SUSPEND

async def hitl_step(state, model, registry, ctx, *, policy) -> AgentState:
    pending = _unanswered_tool_calls(state)   # 最后 assistant 消息中无 tool 回复的 tool_calls
    if pending:
        tc = pending[0]
        name, args = tc 名/参; risk = risk_level(name)
        d = policy(name, args, risk)
        if d == SUSPEND:
            return state.finish("suspended")           # tc 即待审动作,状态已完整
        if d == DENY:
            return state.append(tool_msg(tc.id, f"denied (risk={risk})"))   # 仍 running
        with span("ai_employee.hitl.tool", tool=name, risk=risk):
            result = registry.dispatch(name, args, ctx)
        return state.append(tool_msg(tc.id, result.content))                # 仍 running
    # 无待答 → 调模型
    resp = await chat(model, state.messages, tools=registry.schemas())
    assistant = resp.raw[...]["message"]
    if resp.tool_calls:
        return state.append(assistant)                 # running;tool_calls 成为待答
    return state.append(assistant).finish("done")

async def hitl_run(state, model, registry, ctx, *, policy, max_steps_guard):
    while state.status == "running":
        if state.step >= state.max_steps: return state.finish("exhausted")
        state = await hitl_step(...)        # 注意:每处理一个待答工具 step 不必+1;调模型才 advance
        if state.status == "suspended": return state   # 交给 worker 持久化
    return state
```

> `_unanswered_tool_calls(state)`:取最后一条 `role=assistant` 且含 `tool_calls` 的消息,过滤掉其后已有 `role=tool` 且 `tool_call_id` 匹配的;返回剩余 tool_calls(按出现序)。`step`/`advance` 语义:调模型那次 advance(对齐 max_steps 防失控);处理待答工具不 advance。

### 恢复:把人的决策变成一条 tool 消息,然后继续

```python
def apply_decision(state, *, decision, payload, registry, ctx) -> AgentState:
    tc = _unanswered_tool_calls(state)[0]
    tcid = tc["id"]; name = tc 名
    if decision == "approve":
        result = registry.dispatch(name, tc 原参, ctx); content = result.content
    elif decision == "edit":
        result = registry.dispatch(name, payload["args"], ctx); content = result.content
    elif decision == "reject":
        content = f"[人工拒绝] {payload.get('reason','')}"
    elif decision == "respond":
        content = payload["message"]            # 人替 agent 直接作答,不执行工具
    msgs = state.messages + (tool_msg(tcid, content),)
    return replace(state, messages=msgs, status="running")   # 解挂,继续
```
解挂后再喂回 `hitl_run`,它继续处理剩余待答 tool / 调模型,可能再次挂起(新 inbox 条目)或跑完。

## 存储:ae_inbox 表

```python
class InboxRow(Base):
    __tablename__ = "ae_inbox"
    id: UUID pk
    job_id: UUID | None         # 关联 M1 的 ae_jobs(可空,支持独立 demo)
    employee: Text
    tool_name: Text
    tool_args: JSONB
    risk: Text
    state_json: JSONB           # recovery.dump_state(suspended state)
    status: Text                # "pending" | "resolved"
    decision: Text | None       # approve|edit|reject|respond
    decision_payload: JSONB     # {args:...} | {reason:...} | {message:...}
    created_at / resolved_at: timestamptz
```

## InboxStore

- `suspend(employee, state, *, job_id=None) -> uuid` — 从挂起 state 抽 pending tc,写一条 pending InboxRow(state_json=dump_state)。
- `list_pending(*, employee=None) -> list[InboxRow]`。
- `get(inbox_id) -> InboxRow | None`。
- `resolve(inbox_id, *, decision, payload) -> None` — status→resolved,记 decision/payload/resolved_at。

## 干预写回长期记忆(越用越懂你,喂 M2)

每次 `resolve` 后(或 worker 恢复时),把这次干预写成一条长期记忆,复用 M2a 的 `MemoryStore.insert(kind="fact", embedding=embed_texts(...))`(可被 mem0 召回)或 archival:
- approve:`"审批人批准了 {employee} 的 {tool_name} 操作(参数 {args})"`
- edit:`"审批人把 {tool_name} 的参数从 {old} 改成 {new}"`
- reject:`"审批人拒绝了 {tool_name},原因:{reason}"`
- respond:`"对 {tool_name},审批人直接答复:{message}"`

这样下次同类场景,记忆召回会把"人上次怎么决定"带进 context。M3 用确定性写入(`kind="hitl"` 一律带 embedding),mem0 召回时一并捞。

## 与 worker / CLI 接线

- worker 跑 HITL agent:`hitl_run` 返回 suspended → `InboxStore.suspend(...)` + job→`suspended`(M1 的 ae_jobs status 扩一个值)→ worker 退出。
- CLI Agent Inbox:
  - `anvil-ai-employee inbox list` — 列 pending(id/employee/tool/risk/args 摘要)
  - `inbox approve <id>` / `inbox edit <id> --args '<json>'` / `inbox reject <id> --reason '...'` / `inbox respond <id> --message '...'` — resolve + 写记忆 + 重恢复
  - 恢复:resolve 后 `load_state` → `apply_decision` → `hitl_run` 继续 → 再挂起(新 inbox)或 done;结果落 job。M3 为可测,把"恢复一步"做成纯函数 `resume_from_inbox(inbox_row, registry, ctx, model, policy) -> AgentState`。
- demo 命令:`run-hitl --tool <高风险工具>`(或复用一个带 bash 之类 high-risk 工具的小 registry)跑到挂起,演示 inbox 全流程。

## skills-as-markdown(M2 砍出来的,本期做掉)

- `skills/` 目录放版本化 markdown persona 文件(如 `skills/kb_reporter.md`、`skills/assistant.md`)。
- `load_skill(name) -> str` 读 `skills/<name>.md` 返回 persona 文本;找不到抛清晰错误。
- 把 M1/M2 里硬编码在 Python 的 persona 迁成可选"从文件加载"(保留 Python 默认兜底,不破坏既有)。
- 这是三层记忆的第三层(skills 技能文件)的最小落地。

## 范围边界(M3 明确不做)

| 不做 | 留待 |
|---|---|
| Web 版 Agent Inbox UI(本期纯 CLISP) | 后续 |
| lease 超时自动 reclaim 挂起任务 | 后续 |
| 多级审批 / 审批人鉴权 | 后续 |
| MCP 连接器 | M4 |
| 多员工编队 | M5 |

## 复用与新建

- ✓ 真复用:P3 `permission.risk_level`、`recovery.dump_state/load_state`、`tools.base`(ToolRegistry/dispatch)、gateway `chat`、`AgentState`(+ dataclasses.replace 解挂);M1 的 queue/worker/cli 骨架与 ae_jobs;M2a 的 `MemoryStore.insert`+embedder(干预写记忆);obs span。
- ✗ 新建:`ae_inbox` 表;`HitlDecision`/`HitlPolicy`/`hitl_step`/`hitl_run`/`apply_decision`/`_unanswered_tool_calls`(新模块 `hitl.py`);`InboxStore`;CLI inbox 子命令;`skills/` + `load_skill`;Letta/HITL 兼容性(HITL 与记忆策略正交,可叠加但本期 demo 用简单 registry)。

## 错误处理

- 恢复时 inbox 已 resolved 重复 resolve → 幂等(再调返回已决状态,不重复执行工具)。
- apply_decision 的 edit 参数非法 / 工具执行失败 → 工具返回失败文本当反馈(沿用 ACI 铁律),agent 继续。
- 找不到待审 tc(状态不含未答 tool_call)→ 明确报错(不应发生,挂起时必有)。
- load_skill 文件不存在 → 抛清晰异常(配置问题)。
