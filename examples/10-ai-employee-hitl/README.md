# 10 — AI 员工·Agent Inbox(P4-M3:HITL 防跑飞)

前面 M1-M2 让员工能定时干活、有三层记忆。M3 加上**"防跑飞"的统一范式**:员工跑到**高风险动作**时不直接执行,而是**挂起**等人审批——批准 / 改参 / 拒绝 / 代答;每次人工干预**写回长期记忆**,喂 M2 的记忆系统,越用越懂你该怎么把关。

## 核心机制:挂起 = "最后一条 assistant 消息里尚无 tool 回复的 tool_call"

不需要给状态加"待办动作"字段——**agent 状态本身就完整承载挂起点**:模型提议了一个工具调用(assistant 消息带 tool_calls),但还没有对应的 tool 回复,这个未答的 tool_call 就是待审动作。所以挂起只是 `recovery.dump_state` 存一下,恢复只是把人的决策变成一条 tool 消息注入、继续跑。

`hitl_step` 每步只做一件事:处理一个待答工具(高风险→`finish("suspended")`、否则执行),或调一次模型。干净、可序列化、可恢复——不改 P3 的 `step()`,纯复用 `permission.risk_level` + `recovery` + `tools.base`。

## 四种人工决策(LangGraph Agent Inbox 范式)

| 决策 | 行为 |
|---|---|
| **approve** | 用原参执行该工具 |
| **edit** | 用**改过的参数**执行 |
| **reject** | 不执行,把"被拒+原因"作为反馈喂回 agent,让它重新规划 |
| **respond** | 不执行,人**直接替 agent 作答**(注入一条自定义回复) |

每种决策恢复后,agent 从那条注入的 tool 消息继续往下跑——可能再次挂起(新 inbox 条目)或跑完。

## 跑一遍(需 `ANVIL_DATABASE_URL` + `DEEPSEEK_API_KEY`)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil

# 让员工干一件会触发高风险动作的事 → 它挂起进 Inbox
uv run anvil-ai-employee run-hitl --persona "你是运维助手" --task "删除 /tmp/old 下的日志"
# → [run-hitl] Agent 挂起!高风险动作已进入 Inbox。inbox_id = <id>

# 看待审
uv run anvil-ai-employee inbox list
#   [<id8>] employee=assistant  tool=shell  risk=high  args={'cmd': '...'}

# 四选一处理(都会把这次干预写回长期记忆)
uv run anvil-ai-employee inbox reject  <id> --reason "这台机器不能动"
uv run anvil-ai-employee inbox approve <id>
uv run anvil-ai-employee inbox edit    <id> --args '{"cmd": "ls /tmp/old"}'
uv run anvil-ai-employee inbox respond <id> --message "我手动处理过了"
```

**真实验证**:真 deepseek 收到"删除日志"任务,自己提出了高风险 `shell` 调用,被 HITL harness 挂起进 Inbox;`reject` 后落 resolved 并写了一条干预记忆。整条"挂起→审批→恢复/拒绝→写记忆"在真模型上走通。

## 干预写回长期记忆(喂 M2,越用越懂你)

每次决策写成一条 `kind="hitl"` 的长期记忆(带 embedding,可被 mem0 召回):
- "审批人拒绝了 assistant 的 shell 操作,原因:这台机器不能动。"
- "审批人把 shell 的参数改成 {...}。"

下次类似场景,记忆召回会把"人上次怎么把关的"带进 context——这就是 HITL 与 M2 记忆的衔接点:**人的每次纠偏都沉淀成员工的长期经验**。

## skills 技能文件(三层记忆的第三层,M2 砍出来的)

`skills/*.md` 把 persona/技能外置成版本化 markdown,运行时 `load_skill(name)` 加载(M1/M2 是硬编码在 Python)。这是三层记忆 session/longterm/**skills** 的最后一层落地。

## 复用与新建

- ✓ 真复用:P3 `permission.risk_level`(风险分级)、`recovery.dump_state/load_state`(挂起-恢复)、`tools.base`、gateway;M1 的 queue/worker/CLI 骨架;M2a 的 `MemoryStore.insert`+embedder(干预写记忆)。
- ✗ 新建:`ae_inbox` 表 + `InboxStore`;`hitl.py`(hitl_step/hitl_run/apply_decision)、`inbox_resume.py`、`hitl_memory.py`;CLI inbox 子命令;`skills/` + 加载器。

## 留待

M4(MCP 连接器:令牌服务端托管,agent 拿不到长期凭证)、M5(多员工编队)。Web 版 Inbox UI / lease 超时 reclaim / 多级审批为后续螺旋。
