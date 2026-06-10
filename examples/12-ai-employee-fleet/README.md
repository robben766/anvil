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
