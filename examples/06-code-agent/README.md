# 06 — 自建编码 agent(CA-M1 最小可用循环)

一个 reducer-style while 循环驱动 LLM,用四个工具在 git worktree 隔离内端到端修复 bug。

## 跑 fixture eval(需 DEEPSEEK_API_KEY + ANVIL_DATABASE_URL)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
uv run anvil-code-agent eval --dataset packages/code-agent/src/anvil_code_agent/eval/golden/tasks.jsonl
```

预期:agent 读 calc.py → 发现 `a - b` 应为 `a + b` → edit_file 修正 → run_tests 转绿 → PASS。

## 设计要点

- **agent 即 reducer**:`step(state) -> state'`,纯函数式,易测易恢复
- **编辑可靠性**:SEARCH-REPLACE 唯一匹配护栏,不匹配就报错让模型重读重试(决定可用性)
- **闭环**:run_tests 的失败摘要喂回 context,模型据此继续改
- **隔离**:worktree per task,改动可整体 diff、可丢弃
- **M1 范围**:复用 gateway(tool_use)与 obs(span 追踪);eval-pipeline / SWE-bench 基线集成推迟到 M4

## CA-M2:检索增强

agent 现在能自己定位代码,不靠人喂文件:
- `repo_map` — tree-sitter 抽每个 .py 的函数/类定义,手写 PageRank 把"被引用最多"的文件排前,渲染成预算内的符号地图(抄 Aider 的确定性方案)
- `grep` — 纯 Python regex 全仓搜索(跳过 .git/__pycache__ 等),CI 无外部依赖

设计要点:repo map 把"引用→定义"建成文件图跑 PageRank——定义了被广泛调用符号的文件(枢纽)自然浮到最前,正是 agent 最该先读的地方。

## CA-M3:工程化(可长跑 / 可控 / 可隔离 / 可恢复)

- **上下文压缩**:`estimate_tokens` + `compact` ——超预算时截断老的大工具输出,但绝不删消息/打乱顺序(保住 tool_use 配对的结构正确)。发送前应用,完整历史仍留存供 diff/恢复。
- **权限门**:工具按风险分三档(low/medium/high,未知按 high 安全默认),审批策略是 `(name,args,risk)->bool` 回调;eval 用 auto_approve,交互可换成对 high 风险要人批的门。被拦的工具不执行、转成模型反馈。
- **Docker 沙箱**:`DockerSandbox` 用 docker CLI 起容器、bind-mount workdir,bash 经 `ToolContext.executor` 路由进容器执行——跑任意 shell 才真正进程级隔离(host 子进程是默认回退)。
- **断点恢复**:AgentState 全是普通 dict,`dump_state`/`load_state` 一行落盘/重载,崩溃或主动暂停后可 resume(12-Factor #6)。

## CA-M4:SWE-bench 基线

**本地可复现基线**(三个 bug-fix 任务,后两个多文件):
```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
uv run anvil-code-agent eval --dataset packages/code-agent/src/anvil_code_agent/eval/golden/baseline.jsonl
```
真实跑分:**pass@1 = 100%(3/3)**(deepseek-chat 驱动;calc 4 步、strops 5 步、counter 5 步——后两个多文件任务里 agent 用 repo_map/grep 跨文件定位了 bug)。

**接官方 SWE-bench Lite**(live,拉真实仓):
```bash
# 取官方实例 jsonl(princeton-nlp/SWE-bench_Lite),然后:
uv run anvil-code-agent swebench --dataset swebench_lite.jsonl --limit 5
```
适配器做的事:clone 仓到 base_commit → `git apply` test_patch 并提交(失败测试进 HEAD)→ agent 修 → 跑 FAIL_TO_PASS 判定 pass@1。**刻意不重造官方每实例 Docker 环境构建**——那是 SWE-bench 自己的 harness 范畴;本里程碑触底的是"problem statement → agent → FAIL_TO_PASS 判定"这条评测范式。

## CA-M5:Docker 化(容器内装依赖)

真实 SWE-bench 仓各有各的依赖,host 上 ad-hoc 装会互相打架/装不上。CA-M5 把每个实例关进一个容器:

```bash
# 实例 jsonl 每行可带 "image" 和 "install_cmd";--docker 启用容器隔离
uv run anvil-code-agent swebench --dataset swebench_lite.jsonl --limit 5 --docker
```

容器内流程:起 `image` 容器(默认 python:3.11,带编译工具)→ `install_cmd` 装这个仓的依赖(editable,这样 agent 改源即时生效)→ agent 在容器内读写/跑测试 → 容器内跑 FAIL_TO_PASS。装不上的实例如实报 `docker setup failed`,与"代码没修对"区分开。**刻意不造官方每实例预构建镜像**——临场容器装依赖,够拿到一个诚实的真实 pass@1。

## CA-M6:更深的上下文工程 + 检索

- **摘要压缩 tier**:M3 的 compact 只会截断老工具输出;M6 加一层——超预算时把"中间回合"(系统/任务/最近窗口之外)整段换成一条 LLM 摘要,长任务下 context 不爆且保留要点。关键难点在**不破坏 tool_use 配对**:替换边界对齐到非 tool 消息,孤儿 tool 消息不会出现。
- **符号级 repo map**:M2 排到文件粒度;M6 进一步把每个文件里的符号按"全仓被引次数"排序(被调最多的枢纽函数/类优先),并每文件限 top-K——agent 一眼看到最该关注的符号。
