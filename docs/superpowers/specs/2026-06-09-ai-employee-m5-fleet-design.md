# 产品④ AI 员工 — M5「多员工编队」设计

> 蓝图 §4.5 的集大成收尾。把 M1-M4 打通的**单员工**全链路扩成**一队员工协作**:supervisor 把一个目标 LLM 拆成 N 个独立子任务 → 扇出到 M1 的 PG 队列(每个 job 标 `goal_id` + `employee`)→ 一队 worker 并行处理(复用 M1,SKIP LOCKED 已证无重复领取)→ aggregator 在子任务全终态后综合成最终产出。**最小侵入、独立扇出(无 DAG)、触底"supervisor→fan-out→aggregate"核心范式即走**。

## 目标(一句话)

`team run --goal "<目标>"` → supervisor 拆解并扇出子任务给不同 employee → fleet workers 并行跑完 → aggregator 综合 → `ae_goals.result` 落最终产出;全程复用 M1 队列 + M1 worker + P3 harness + guard.structured_chat。

## 核心机制一:supervisor 拆解 + 扇出

```python
# fleet/supervisor.py
@dataclass
class SubTask:
    employee: str        # 指派给哪个员工(必须在 EMPLOYEES 注册表里)
    task: str            # 该员工要做的具体子任务描述

async def decompose(goal: str, *, model: str, employees: list[str]) -> list[SubTask]:
    """用 guard.structured_chat(json_object)把 goal 拆成子任务列表。
    prompt 含字面量 'json';约束每个 subtask 的 employee 必须 ∈ employees;
    解析失败/employee 非法 → 过滤掉非法项;全空则兜底成单条交给第一个 employee。"""

async def fan_out(
    session_factory, *, goal_id, subtasks: list[SubTask]
) -> list[uuid.UUID]:
    """对每个 SubTask 调 M1 的 enqueue(skill=employee 对应技能, payload={'task':...},
    goal_id=goal_id, employee=subtask.employee)。返回 job_id 列表。"""
```

- 拆解 prompt:给定 goal + 可用 employees 及其能力简介,要求产出 `{"subtasks":[{"employee":"...","task":"..."}]}`。
- `structured_chat` 已在 P2/guard 验证(json_object 模式,prompt 必含 "json")。
- **健壮性**:LLM 返回的 employee 不在注册表 → 丢弃该条;subtasks 全空 → 兜底成 `[SubTask(employees[0], goal)]`(至少跑一个,不空转)。

## 核心机制二:fleet —— 直接复用 M1 队列,worker 泛化

M1 的 `worker.run_once` 已经 `claim_one`(SKIP LOCKED)→ 跑 agent → complete/fail,多进程并行抢 job 无重复领取(M1 用例已证)。**M5 唯一改动:worker 按 `job.employee` 选 persona/registry,而非硬编码 kb_reporter**。

```python
# fleet/team.py
@dataclass
class Employee:
    name: str
    description: str                  # 供 supervisor 拆解时了解其能力
    persona: str
    build_registry: Callable          # (EmployeeContext) -> ToolRegistry
    task_prompt: str = "现在开始执行你被指派的任务。"

EMPLOYEES: dict[str, Employee] = {
    "kb_reporter": Employee(...kb_digest 的 persona/registry...),
    "researcher":  Employee(...新增 demo 员工:用 kb_search 工具做主题调研,产调研纪要...),
}
```

```python
# worker.py 改动(最小):
#   skill 字段语义保持(M1 的 "kb_digest" 仍可跑);新增 employee 维度
#   employee = job.employee or "kb_reporter"(fallback 保 M1 行为)
#   emp = EMPLOYEES.get(employee);取 emp.persona / emp.build_registry / emp.task_prompt
#   job 的 task 优先取 job.payload['task'](编队子任务),否则用 emp.task_prompt(M1 周报)
```

> M1 的 `SKILLS = {"kb_digest": ...}` 与 M5 的 `EMPLOYEES` 关系:M1 用 `skill` 选行为;M5 用 `employee` 选行为。为不破坏 M1,worker 先看 `job.employee`(新),没有就回落到老的 `skill`→SKILLS 路径。两条路径都保留,M1 既有测试不动。

## 核心机制三:aggregator —— 子任务全终态后综合

```python
# fleet/aggregator.py
async def children_terminal(session_factory, goal_id) -> bool:
    """该 goal 的所有子 job 是否都 done 或 failed(无 pending/running)。"""

async def aggregate(session_factory, goal_id, *, model: str) -> str | None:
    """若 children_terminal:收集各子 job 的 (employee, task, result/error) →
    guard.structured_chat 或普通 chat 综合成最终 markdown → 写 ae_goals[goal_id].result,
    status='done'。返回最终产出。若还有未完成子任务 → 返回 None(不综合)。"""
```

- 综合 prompt:把各员工的子任务与产出列出,要求合并成一份连贯的最终交付物。
- 失败的子任务也带进综合(标注 `[失败: error]`),让最终产出诚实反映哪部分没成(ACI 精神延伸到编队层)。
- aggregator 可由 CLI `team status` 触发(轮询),或独立调用;不引入新后台进程。

## 存储:ae_goals 表 + JobRow 两列

```python
class GoalRow(Base):
    __tablename__ = "ae_goals"
    id: UUID pk
    objective: Text                   # 用户给的高层目标
    status: Text                      # "pending" | "running" | "done"
    result: Text | None               # aggregator 综合后的最终产出
    created_at / finished_at: timestamptz

# JobRow 加两列(都 nullable,M1-M4 单 job 路径不破):
#   goal_id: UUID | None              # 关联 ae_goals(独立 job 为 None)
#   employee: Text | None             # 指派的员工名(None → worker fallback "kb_reporter")
```

## 与 CLI 接线

- `anvil-ai-employee team run --goal "<目标>" [--model ...]` — 建 ae_goals(status=running)→ supervisor.decompose → fan_out 扇出子 job → 打印 goal_id + 子任务清单。
- `anvil-ai-employee team status --goal <id>` — 列各子 job 状态;若全终态则调 aggregate 综合并打印最终产出。
- `anvil-ai-employee work --loop`(M1 已有)= fleet worker:现在能跑任意 employee 的 job(按 job.employee 选 registry)。多开几个 `work` 进程 = 多 worker 并行编队。

## Demo(真实验证)

1. `team run --goal "调研'向量检索'并产一份知识库周报"` → supervisor 拆成 `[{researcher, "调研向量检索"}, {kb_reporter, "产周报"}]`,扇出 2 个子 job。
2. 起 worker(可多开)→ 并行跑:researcher 用 kb_search 产调研纪要、kb_reporter 产周报,各自 complete。
3. `team status --goal <id>` → 两子任务 done → aggregate 综合成最终交付物,落 ae_goals.result。
4. 真 deepseek 验证整条 decompose→fan-out→并行 work→aggregate 走通;含一个子任务故意失败的诚实综合用例(综合产出标注失败部分)。

## 错误处理

- supervisor 拆解返回非法/空 → 过滤 + 兜底单任务(不空转)。
- 子 job 失败(M1 worker 已 fail 并记 error)→ 不阻塞 goal;aggregator 把失败也纳入综合并标注。
- aggregate 在子任务未全终态时调用 → 返回 None,不写 result(幂等:重复调直到全终态)。
- goal 无子任务(扇出 0 条)→ 不应发生(supervisor 兜底);若发生,status 置 done、result 记"无子任务"。

## 范围边界(M5 明确不做)

| 不做 | 留待螺旋 |
|---|---|
| 子任务 DAG 依赖(本期独立扇出) | 后续 |
| 员工间消息 / 协商 / 移交 | 后续 |
| 动态 worker 扩缩容 / 优先级调度 | 后续 |
| 陪审团(P2)择优综合(本期简单合并) | 后续(P2 council 现成可接) |
| goal 级 HITL 审批(子任务级已由 M3 覆盖) | 后续 |

## 复用与新建

- ✓ 真复用:M1 `scheduler/queue`(enqueue/claim_one SKIP LOCKED/complete/fail,语义不变)+ `worker.run_once` 骨架 + P3 `harness.run` + `EmployeeContext`/`tools`;`anvil_guard.structured_chat`(拆解+综合);obs span;M2 记忆 / M3 HITL / M4 MCP 员工天然可作编队成员(各 employee 自带能力,M5 不改它们)。
- ✗ 新建:`ae_goals` 表 + `JobRow.goal_id/employee` 两列;`fleet/`(supervisor/aggregator/team);worker 的 employee 泛化分支;CLI `team` 子命令;example 12。

## 测试策略(TDD)

- `db`:ae_goals CRUD + JobRow 新列默认值(真 PG@5434);M1 既有 db 测试不回归。
- `supervisor.decompose`:mock gateway(respx)返合法/非法/空 JSON → 过滤非法 employee、空兜底单任务。
- `supervisor.fan_out`:扇出 N 子 job,各带 goal_id+employee,真 PG。
- `team`:EMPLOYEES 注册表至少 2 员工,各有 persona/registry builder。
- `worker`:job 带 employee=researcher → 选 researcher 的 registry(respx mock gateway 跑到 complete);job 无 employee → fallback kb_reporter(M1 行为不变,既有用例绿)。
- `aggregator`:children_terminal(混合 done/failed/pending)判定;aggregate 未全终态→None、全终态→综合写 result(含失败子任务标注);幂等。
- 集成:decompose→fan_out→worker×并行→aggregate 端到端(respx mock 或 live marker)。
- 全仓 `pytest -m "not live"` 必绿(沿用 M1-M4 教训:全仓收集防文件名/fixture 撞车)。
