# 产品④ AI 员工 — M1「知识库周报员」设计

> 蓝图章节:`anvil-notes/AI工程全景调研与四产品落地蓝图.md` §4.5。本 spec 仅覆盖 **P4-M1**:单触发(cron)+ 单技能(知识库周报员)的最小垂直骨架。三层记忆全量 / Agent Inbox / MCP / 多员工编队分别留给 M2~M5。

## 目标(一句话)

cron 定时唤醒一个"知识库周报员":它在沙箱里跑 P3 的 agent 循环,用 P1 知识库的检索工具读"上次报告之后新入库"的内容,产出一份结构化摘要,并把"报告覆盖到哪个时间点"写回长期记忆——下次不重复。

## 为什么是这个 MVP

一个垂直切片同时串起四样既有/新建能力,证明"前三个产品能合体":

- **cron 调度 + PG 队列**(新建,本产品骨架)
- **P3 harness 循环**(复用 `anvil_code_agent.harness.loop` + `tools.base` 的 `@tool`/`ToolRegistry` 协议)
- **P1 知识库**(复用 `anvil_kb` 的 `Retriever` 与 `DocumentRow`)当"知识"
- **最小长期记忆**(新建,本产品灵魂的雏形)

严格守"单技能单触发":记忆只做"报告标记",不做 mem0 抽取 / 向量召回 / Letta 换页(那是 M2)。

## 架构

```
packages/ai-employee/  (anvil_ai_employee)
├─ db.py                SQLAlchemy 异步 Base + 三张表(见下)+ Alembic 迁移
├─ scheduler/
│  ├─ trigger.py        Trigger 协议 + CronTrigger(croniter 解析,读 ae_schedules)
│  └─ queue.py          PG 队列:enqueue / claim_one(FOR UPDATE SKIP LOCKED) / complete / fail
├─ memory/
│  └─ store.py          MemoryStore:write_marker / last_marker / recent(按 employee+kind+时间)
├─ skills/
│  └─ kb_digest.py      周报员技能:system prompt(persona)+ build_registry(EmployeeContext)
├─ tools.py             周报员的 @tool 们:kb_recent / kb_search / recall_marker / submit_report
├─ worker.py            run_once:claim → 按 skill 派发 → 跑 P3 loop → 落库 result + 写记忆标记
└─ cli.py               anvil-ai-employee: add-schedule / tick / work / run-now / report
```

### 数据模型(PG,三张表,全 SQLAlchemy 异步 + Alembic)

**`ae_schedules`** — 定时计划
- `id` UUID pk, `name` text unique, `cron_expr` text, `skill` text, `payload` JSONB(给技能的参数)
- `next_run_at` timestamptz, `enabled` bool default true, `created_at` timestamptz

**`ae_jobs`** — 任务队列(PG 队列的载体)
- `id` UUID pk, `schedule_id` UUID null(ad-hoc 为 null), `skill` text, `payload` JSONB
- `status` text in {pending, running, done, failed} default pending
- `result` text null, `error` text null, `locked_by` text null
- `created_at` / `started_at` null / `finished_at` null timestamptz

**`ae_memories`** — 最小长期记忆(M1 只存"报告标记";M2 扩为三层记忆)
- `id` UUID pk, `employee` text(如 "kb_reporter"), `kind` text(M1 用 "report_marker")
- `content` text(JSON 字符串:{covered_until: iso, summary_head: str}), `created_at` timestamptz
- M1 召回方式 = 按 `employee + kind` 取最新一条(纯 recency,无向量)。**embedding/向量召回是 M2,不入本表本期。**

### 控制流

```
[系统 cron 或 ai-employee tick --loop]
        │ 每分钟一拍
        ▼
CronTrigger.due(now)  ── 读 ae_schedules,croniter 算出到点的 → 生成 job spec,推进 next_run_at
        │
        ▼  enqueue → ae_jobs(status=pending)
        ┊
[ai-employee work --loop]  (一个或多个 worker 进程)
        │
        ▼
queue.claim_one(worker_id)  ── SELECT … FOR UPDATE SKIP LOCKED LIMIT 1 → status=running, locked_by
        │
        ▼
worker.dispatch(skill="kb_digest")
        │   build EmployeeContext(db, employee="kb_reporter")
        │   registry = kb_digest.build_registry(ctx)   # kb_recent/kb_search/recall_marker/submit_report
        │   state0 = AgentState.new(messages=[system=persona, user=任务], workdir=临时)
        ▼
P3  run(state0, model, registry, toolctx)   ← 复用 harness.loop,进程内 host executor
        │   agent: recall_marker → kb_recent(since) → [kb_search…] → submit_report(markdown)
        ▼
submit_report 工具内:把 markdown 存进 job.result + 写 ae_memories(report_marker, covered_until)
        │
        ▼
queue.complete(job)  → status=done, finished_at
```

整条链路包在一个 obs span 里(复用 `anvil_obs`),失败走 `queue.fail`(status=failed, error)。

### 工具(周报员的 ACI,P3 `@tool` 风格)

这些工具不依赖 P3 的 `ToolContext`(那是给文件/shell 用的);它们在 `build_registry(ctx)` 时闭包捕获 `EmployeeContext`(db session 工厂 + employee 名),自己访问 DB / 调 P1。

- `recall_marker() -> str` — 读本员工最新 `report_marker`,返回上次覆盖到的 iso 时间(没有则返回"从未报告,取最近 7 天")。
- `kb_recent(since_iso: str) -> str` — 查 `kb_documents` WHERE `created_at > since`,返回每篇的 title/source/created_at + content 前 N 字预览(按时间升序)。这是"新入库了什么"。
- `kb_search(query: str, k: int = 5) -> str` — 包 P1 `Retriever.retrieve(query, k)`,返回 top-k chunk 文本。用于对某主题深读。
- `submit_report(markdown: str, covered_until_iso: str) -> str` — **终止工具**:把 markdown 写进当前 job.result,并写一条 `report_marker` 记忆(covered_until + summary 头部)。返回 "ok"。agent 调完它即视为完成。

### 技能 / persona(`skills/kb_digest.py`)

system prompt 要点(中文):"你是知识库周报员。步骤:① `recall_marker` 拿到上次覆盖到的时间点;② `kb_recent(since)` 列出此后新入库的文档;③ 若某主题值得展开,用 `kb_search` 深读;④ 写一份结构化中文摘要(按主题分组,每条标注来源 source);⑤ 调 `submit_report(markdown, covered_until)` 提交,covered_until 取你这次见过的最大 created_at。若 `kb_recent` 为空,提交一句'本期无新增'并把 covered_until 设为当前时间。"

### CLI(`anvil-ai-employee`)

- `add-schedule --name 周报 --cron "0 9 * * 1" --skill kb_digest` — 建定时计划(每周一 9 点)
- `tick [--loop]` — 跑一拍调度:把到点的计划入队(系统 cron 调 `tick`,或 `--loop` 自带 ticker)
- `work [--loop]` — 起 worker 消费队列
- `run-now --skill kb_digest` — 立即入队一个 ad-hoc job(证明 on-demand 触发的接缝,也是测试便利)
- `report --job <id>` — 打印某 job 的 result

## 错误处理

- 工具失败永不抛异常崩循环(沿用 P3 铁律):读不到库 / 查询出错 → 返回可读失败串当反馈。
- worker 跑 P3 loop 抛异常 → `queue.fail(job, error)`,status=failed,不影响其他 job。
- `claim_one` 用 `SKIP LOCKED` 保证多 worker 不抢同一 job;crash 的 job 留 running(M1 不做自动 reclaim,M3 再补 lease 超时)。
- cron 推进 `next_run_at` 与 enqueue 在**同一事务**,避免重复入队或漏触发。

## 测试策略(TDD,真 PG@5434)

- **queue**:多"worker"并发 claim 同一批 pending,断言无重复领取(SKIP LOCKED 行为);complete/fail 状态流转。
- **CronTrigger**:给定 cron 表达式 + 固定 now(注入,不用 `Date.now`),断言 due 集合正确、next_run_at 正确推进、未到点不触发。
- **MemoryStore**:write_marker 后 last_marker 取回最新;多员工隔离。
- **tools**:kb_recent 对预置 kb_documents 按 since 过滤正确;submit_report 落 result + 写记忆标记。
- **worker run_once**(集成,可用假 model / 录制工具序列):一个 pending job → 跑完 → status=done、result 非空、记忆标记写入。
- **端到端 live**(`-m live`,手动):真 deepseek 驱动,预置几篇 kb 文档 → run-now → 周报员产出真摘要、covered_until 正确。

## 范围边界(M1 明确不做)

| 不做 | 留给 |
|---|---|
| mem0 抽取(ADD/UPDATE/DELETE)+ 向量召回 + Letta 自主换页 | M2 三层记忆 |
| Agent Inbox / 高风险挂起 / 人工 approve-edit-reject | M3 HITL |
| MCP 连接器 / OAuth 令牌服务端托管 | M4 |
| 多员工编队 / 员工间协作 | M5 |
| Docker 沙箱隔离 worker(只读 KB 不需要) | 接"代码值守员"时 |
| webhook 触发(Trigger 抽象已留口,不实现) | 后续 |

## 复用与依赖

- 复用:`anvil_code_agent.harness.loop`(run/step)、`anvil_code_agent.tools.base`(Tool/ToolResult/ToolContext/ToolRegistry/@tool)、`anvil_kb.retrieve.Retriever`、`anvil_kb.db.DocumentRow`、`anvil_obs`(span)、`anvil_gateway`(LLM)。
- 新增依赖:`croniter`(cron 解析)。
- 存储:PG@5434(`ANVIL_DATABASE_URL`),SQLAlchemy 异步 + Alembic,与 anvil 全栈一致。无 SQLite、无 Redis。
- 根 `pyproject.toml` `members` 增 `packages/ai-employee`。
