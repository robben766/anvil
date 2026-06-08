# P3 — AI 编码团队(自建编码 agent harness)设计文档

> 状态:已批准(2026-06-08 头脑风暴定稿)
> 对标:Claude Code / OpenHands 的 agent harness
> 范围口径:**完整实现单 agent harness(步骤 1-5),分里程碑交付;多 agent team 层归 P3-B,暂停。**

## 1. 目标与背景

P3 的核心是**自研一个编码 agent harness**,吃透"Claude Code 内部到底怎么转"这层范式——
agentic 循环、tool_use 协议全往返、ACI(Agent-Computer Interface)工具设计、上下文工程、闭环验证。

遵循 **98.4/1.6 法则**:绝大部分工作量在 harness(控制流/工具/反馈闭环),不在 prompt。
遵循 anvil **深度学习优先**:核心循环与工具自研造轮子,框架(LangGraph/Aider/OpenHands)只当教材读,不引入。

**与 P3-B 的关系**:本设计只做"单个强 agent"。B(编排多个现成 CLI agent 协作)已论证技术可行但暂停;
将来本 harness 产出的自研 agent 正好能当 B 编排里的一个"座位"。多 agent team 层不在本设计范围。

### 1.1 成功标准(看得出效果)

终态 = 一个能在**真实仓库**里端到端"读码 → 改 → 跑测试 → 修到绿"修复 bug 的自研 agent,
并在 **SWE-bench Lite 子集**上跑出 pass@1 基线数字、进 CI。绝不是 toy demo。

## 2. 范围

### 2.1 在范围内(完整单 agent harness,步骤 1-5)

1. **最小 ReAct 循环** —— while 循环 + message 数组 + tool_use 协议全往返
2. **可靠编辑** —— SEARCH-REPLACE diff 格式 + 护栏 + 重试 + lint/test 闭环反馈
3. **repo map** —— tree-sitter + PageRank(抄 Aider 的确定性方案)+ agentic grep
4. **工程化** —— context 压缩流水线 + 权限审批门(三档)+ Docker 沙箱 + 断点恢复
5. **评测基线** —— SWE-bench Lite 子集,pass@1,进 CI

### 2.2 不在范围内(明确后置)

- **多 agent team 层**(planner / 并行 workers / reviewer 编队)—— 这是 P3-B 的领域,暂停。
- **IDE/Web 前端** —— 本产品是 CLI + 库;可视化后置。
- **非 Python 语言支持** —— 首期 Python 优先(SWE-bench Lite 是 Python,anvil 本身也是)。

## 3. 架构

### 3.1 核心模型:agent 即无状态 reducer

循环用 Claude Code 模式(**while 循环 + message 数组,不要 DAG**)。一步 = 一次 reduce:

```
step(state) -> state':
  1. 把 state.messages 连同 tool schema 发给 gateway.chat(tools=...)
  2. 收到 assistant 消息:
     - 若 finish_reason == "tool_calls":逐个执行工具 → 把每个结果作为
       {role:"tool", tool_call_id, content} 追加进 messages → 继续循环
     - 若是普通文本 / 模型声明完成:进入收尾(跑最终验证)→ 终止
  3. 守护:超过 max_steps / token 预算 / 连续无进展 → 安全终止
```

state 是不可变快照:`AgentState{messages, step, budget, workdir, status}`。
`step()` 是纯函数式 reducer:`(state, action) -> new_state`,便于测试与断点恢复。

### 3.2 包结构(`packages/code-agent/`,与 P1 `packages/kb` 平级)

```
packages/code-agent/
├─ pyproject.toml            # name=anvil-code-agent;deps: anvil-gateway/obs/eval/guard
└─ src/anvil_code_agent/
   ├─ __init__.py
   ├─ state.py              # AgentState 数据类 + 不可变更新
   ├─ harness/
   │  ├─ loop.py            # run(state, tools) while 循环;reducer step()
   │  ├─ context.py         # token 预算管理(压缩流水线 M3 接)
   │  ├─ permission.py      # 工具风险分级 + 审批门(M3)
   │  └─ recovery.py        # 错误压进 context + 断点恢复(M3)
   ├─ tools/
   │  ├─ base.py            # Tool 协议:name/schema/run();注册表
   │  ├─ fs.py              # read / edit(SEARCH-REPLACE + 护栏)
   │  ├─ shell.py           # bash(超时 + 输出截断)
   │  ├─ search.py          # repo map(tree-sitter+PageRank)+ grep(M2)
   │  └─ verify.py          # lint/test 运行器(闭环反馈)
   ├─ sandbox.py            # git worktree(M1)→ Docker(M3)
   ├─ cli.py                # anvil-code-agent solve <task> / eval <dataset>
   └─ eval/
      ├─ task.py            # Task 数据类 + 加载器
      ├─ runner.py          # 跑 agent → 验证 → pass@k 指标
      └─ golden/
         ├─ fixtures/       # M1 手造 bug-fix 任务(小仓 + 失败测试)
         └─ swebench_lite/  # M4 SWE-bench Lite 子集清单
```

### 3.3 工具协议(ACI 设计是重点)

SWE-agent 的核心洞察:**工具接口质量比模型本身更决定 agent 成败**。每个工具:

```python
class Tool(Protocol):
    name: str
    schema: dict          # OpenAI function-calling JSON schema
    def run(self, args: dict, ctx: ToolContext) -> ToolResult: ...

@dataclass
class ToolResult:
    content: str          # 喂回模型的文本(失败也要可读、可指导下一步)
    ok: bool
    truncated: bool = False
```

- `ToolContext` 携带 `workdir`(沙箱内路径)、超时、输出上限。
- 工具**永远返回可读结果**,即使失败——失败信息本身是给模型的反馈(12-Factor #9:错误压进 context)。

#### 核心工具

| 工具 | 入参 | 关键设计 |
|---|---|---|
| `read_file` | path, [range] | 带行号返回;超长截断 + 提示 |
| `edit_file` | path, search, replace | **SEARCH-REPLACE diff**:search 块必须在文件中**唯一精确匹配**,否则报错让模型重试;护栏:不存在/多处匹配/空 search 全部拒绝并给明确原因 |
| `bash` | cmd | 子进程 + **超时**(默认 120s)+ **输出截断**(默认 4KB,头尾保留);沙箱内执行 |
| `repo_map`(M2) | [focus] | tree-sitter 抽符号 + PageRank 排关键文件,返回精简代码地图 |
| `grep`(M2) | pattern | agentic ripgrep 封装 |
| `run_tests`(verify) | [target] | 跑 pytest/lint,**结构化返回 pass/fail + 失败摘要**,是闭环命脉 |

### 3.4 沙箱与隔离

- **M1-M2:git worktree** —— 每个任务在目标仓的独立 worktree 里跑,agent 的改动隔离、可整体 diff、可丢弃。轻、够用。
- **M3:Docker 容器 per session** —— 进程级隔离,跑任意 bash 才真正安全;worktree 挂进容器。
- bash 工具在 M1 即带超时 + 输出截断 + 禁网(可配)作为最低护栏。

### 3.5 数据流(一个任务的一生)

```
Task(repo, 问题描述, 验证命令)
   │
   ▼ sandbox.create() → 独立 worktree/容器
AgentState 初始化(system prompt + 任务描述 + 工具 schema)
   │
   ▼ harness.loop.run() —— reducer 循环
   ├─ 模型决定 read_file / grep / repo_map → 定位
   ├─ 模型 edit_file(SEARCH-REPLACE)→ 改
   ├─ 模型 run_tests → 看反馈 → 没绿就接着改(闭环)
   └─ 模型声明完成 / 守护触发 → 终止
   │
   ▼ runner 跑验证命令(任务自带的 ground-truth 测试)
   └─ pass / fail 记入 eval 指标;worktree diff 留痕
```

全程每一步 tool 调用用 `anvil_obs.span` 追踪(看 agent 怎么想的、卡在哪)。

## 4. 里程碑划分

每个里程碑都是**能跑的真 agent**,不是半截子。每个里程碑产出一个独立的 spec→plan→执行循环。

### CA-M1 最小可用循环
**交付**:`loop.py` reducer 循环 + `fs.read/edit` + `shell.bash` + `verify.run_tests` 闭环 +
`sandbox` worktree + 小型 fixture bug-fix 任务集 + runner(pass 率)。
**验收**:能端到端修复 fixture 里的真 bug(读码→改→跑测试→修到绿),eval 给出 pass 率数字。
**触底**:agent loop + tool_use 协议 + ACI 编辑工具 + 闭环反馈。

### CA-M2 检索增强
**交付**:`search.py` repo map(tree-sitter + PageRank)+ agentic grep,接进工具集。
**验收**:在多文件中等仓库里,agent **不靠人喂文件**也能自己定位到要改的代码;fixture 扩到多文件任务。

### CA-M3 工程化
**交付**:`context.py` 分级压缩流水线 + token 预算;`permission.py` 三档审批门(每步/计划/PR);
`sandbox.py` Docker 后端;`recovery.py` 断点恢复。
**验收**:长任务不爆 context;高风险工具按档拦截审批;容器隔离跑任意 bash;中断可续。

### CA-M4 评测基线
**交付**:SWE-bench Lite 子集清单 + 适配器(拉取/构建/验证)+ pass@1 指标,进 CI(标 live/slow)。
**验收**:跑出 pass@1 基线数字并写进 example/README;CI 有一条冒烟任务常绿。

## 5. 底座复用

| 底座 | 用法 |
|---|---|
| **anvil-gateway** | `chat(model, messages, tools=...)` —— LLM 调用 + tool_use 往返 + 记账(已验证支持完整往返) |
| **anvil-obs** | `span(name, **attrs)` —— 追踪循环每一步 tool 调用 |
| **anvil-eval** | 复用 pass@k / 指标基建;runner 产出可校准数字 |
| **anvil-guard** | 可选:`edit_file`/`bash` 的入参做注入/越界校验(M3 权限门复用) |

## 6. 错误处理

- **工具失败 = 给模型的反馈**,不是异常上抛。所有工具返回 `ToolResult(ok=False, content=可读原因)`,模型据此重试(12-Factor #9)。
- **edit 不匹配**:SEARCH 块未唯一匹配 → 明确告知"未找到/多处匹配",要求模型重读文件再试。
- **循环守护**:max_steps、token 预算、连续 N 步无文件变更 → 安全终止并标记 `status=exhausted`。
- **沙箱崩溃 / 超时**:捕获 → 记 eval 为 fail,不污染其他任务(runner 逐任务 try/except 隔离)。

## 7. 测试策略(TDD + eval 先行)

- **单元(mock,无 key,走 `not live`)**:reducer step 状态转移、SEARCH-REPLACE 匹配/护栏边界、bash 超时与截断、tool schema 往返(respx mock gateway)、repo map 排序对照、pass@k 计算手算对照。
- **集成(fixture)**:在内置小仓上跑完整循环,断言能把失败测试修绿(用确定性 mock 或便宜真模型,标记)。
- **live(标 `@pytest.mark.live`,手动 / CI 慢道)**:真模型驱动跑 fixture / SWE-bench Lite 冒烟。
- **eval 先行铁律**:没有 pass 率数字不算完成;M4 起 eval 进 CI。

## 8. 已定决策

1. **沙箱**:worktree 先行(M1)→ Docker 加固(M3)。
2. **eval**:fixture 手造任务(M1)→ SWE-bench Lite 子集(M4)。
3. **驱动模型**:deepseek 默认(代码强 + 支持 tool_use,走 gateway 可配),M4 可换更强对比。
4. **语言**:Python 优先。

## 9. 风险

| 风险 | 缓解 |
|---|---|
| 模型 tool_use 能力参差(deepseek/qwen 对 function-calling 支持度) | M1 先用确定性 fixture + 验证往返;模型可配,弱则换 |
| SEARCH-REPLACE 编辑不可靠(模型常给不精确块) | 护栏 + 清晰报错驱动重试;这是"决定可用性的关键",M1 重点打磨 |
| SWE-bench Lite 环境重(每任务 Docker 镜像) | M4 才上,先取极小子集 + 冒烟;不进默认 `not live` |
| 范围膨胀(忍不住做多 agent) | 边界已划死:多 agent = B,本设计只单 agent |

## 10. 吃透原理清单(学习目标)

ReAct 循环、tool_use 协议全往返、SEARCH-REPLACE 编辑可靠性、tree-sitter AST、PageRank、
上下文压缩策略、git worktree / Docker 隔离、12-Factor Agents(错误压进 context、断点恢复)。
