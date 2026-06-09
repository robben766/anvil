# 07 — AI 员工(P4 集大成 · M1「知识库周报员」)

P4 是四产品的最后一块拼图:把前三个产品(① 知识库、② 评测、③ 编码 agent)的能力拧成一个"长期在岗、定时干活、越用越懂你"的 AI 员工。**M1 是最小垂直骨架**:cron 定时唤醒一个"知识库周报员",在进程内跑 P3 的 agent 循环,用 P1 知识库的检索工具读"上次报告之后新入库"的内容,产出结构化摘要,并把"覆盖到哪个时间点"写回长期记忆——下次不重复。

## 为什么这一个 MVP 就是"集大成"

一个垂直切片同时串起四样能力:

```
[cron / ai-employee tick]                         ← 调度(新建,本产品骨架)
        │ 到点 → 入队
        ▼
ae_jobs (PG 队列, SELECT … FOR UPDATE SKIP LOCKED) ← 零新基建,与 anvil 全 PG 一致
        │ worker 抢任务
        ▼
worker → AgentState.new(persona=周报员) 
        → anvil_code_agent.harness.run(...)        ← 复用 P3 harness 的 reducer 循环
              ├─ recall_marker()    上次报告到哪了 ← 最小长期记忆(本产品灵魂的雏形)
              ├─ kb_recent(since)   新入库了什么    ← 复用 P1 知识库 DocumentRow
              ├─ kb_search(query)   某主题深读      ← 复用 P1 Retriever(dense)
              └─ submit_report(md)  提交 + 写标记
        ▼
ae_jobs.result = 周报 markdown ; ae_memories += report_marker(covered_until)
```

复用清单:**P3 harness**(`anvil_code_agent.harness.loop` + `@tool` 协议)、**P1 知识库**(`anvil_kb` 的 `Retriever`/`DocumentRow`)、**gateway**(LLM tool_use 往返)、**obs**(每个 job 一个 span)。新建的只有:PG 队列、cron 触发器、最小长期记忆。

## 跑一遍(需 `ANVIL_DATABASE_URL` + `DEEPSEEK_API_KEY`)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil

# 1. 先灌点知识库内容(复用 P1;golden 语料是 3 篇虚构保险产品文档)
uv run anvil-kb ingest packages/kb/golden/corpus/*.md

# 2. 立即派一个周报任务(ad-hoc,证明 on-demand 入队接缝)
uv run anvil-ai-employee run-now --skill kb_digest        # → 打印 job <id>

# 3. 起 worker 跑一次:claim → 跑 P3 循环 → 产出周报 → 写记忆标记
uv run anvil-ai-employee work

# 4. 看周报
uv run anvil-ai-employee report --job <id>
```

定时模式:

```bash
# 每周一 9 点
uv run anvil-ai-employee add-schedule --name 周报 --cron "0 9 * * 1" --skill kb_digest
uv run anvil-ai-employee tick --loop &   # ticker:到点把计划入队(或交给系统 cron 周期调 tick)
uv run anvil-ai-employee work --loop &   # worker:常驻消费队列
```

第二次跑时,`recall_marker` 会读到上次的 `covered_until`,`kb_recent` 只列此后的新增——这就是"记得自己干到哪了、不重复"的最小记忆闭环。

## M1 的边界(刻意不做,留给后续里程碑)

| 不做 | 留给 |
|---|---|
| 完整三层记忆(mem0 抽取 ADD/UPDATE/DELETE + 向量召回 + Letta 自主换页) | M2 |
| Agent Inbox / 高风险动作挂起 / 人工 approve-edit-reject | M3 HITL |
| MCP 连接器 / OAuth 令牌服务端托管(邮件/IM/日历) | M4 |
| 多员工编队 / 员工间协作 | M5 |
| Docker 沙箱隔离 worker(只读知识库不需要) | 接"代码值守员"时 |

M1 记忆只做"报告标记"(记住上次覆盖到的时间点),足够证明"越用越懂"的雏形;完整三层记忆是 M2 的主题。

## 测试

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test
uv run pytest packages/ai-employee -q
```

队列的并发用例(8 worker 抢 5 job 无重复领取)证明了 `SKIP LOCKED` 语义;worker 集成用例用 respx 录一段工具序列,证明 worker 端到端驱动 P3 循环把 job 跑到 done。live 冒烟(真 deepseek 产出周报)需 `DEEPSEEK_API_KEY`。
