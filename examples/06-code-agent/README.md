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
