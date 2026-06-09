# 09 — AI 员工·三层记忆(P4-M2b:Letta/MemGPT 哲学)

M2a 给员工装了 **mem0 哲学**的记忆(编排器在背后管,agent 不知记忆存在)。M2b 装上**对照的另一种哲学——Letta/MemGPT 自管式记忆**:agent **自己调工具**管理记忆。两种并排,就是为了吃透 P4 记忆这一层最核心的分野——**谁来决定记什么、什么时候换页:编排器,还是 agent 自己?**

## mem0 vs Letta:同一件事,两种世界观

| | **mem0(M2a,托管式)** | **Letta(M2b,自管式)** |
|---|---|---|
| 谁管记忆 | 编排器(召回 + 抽取调和) | **agent 自己**(回合内调工具) |
| agent 手里有记忆工具吗 | 没有(纯对话) | **有 5 个** |
| 每轮 system 注入 | 向量召回的事实 | 常驻 **core memory 块** |
| 一轮结束后 | 编排器 LLM 抽取→ADD/UPDATE/DELETE | **什么都不做**(agent 已在回合内自己改完) |
| 上下文满了 | 不涉及(记忆是抽取的离散事实) | **self-paging**:提示 agent 存记忆 + 递归摘要换页 |

## MemGPT 三层记忆(本里程碑完整实现)

```
┌─ core memory ──── ae_core_blocks(persona/human 块,常驻 system,有字符上限)
│   agent 工具:core_memory_append / core_memory_replace
├─ recall memory ── ae_sessions.messages(全量对话史,self-paging 压缩活动上下文后仍可查)
│   agent 工具:conversation_search
└─ archival memory ─ ae_memories kind=archival(向量化)
    agent 工具:archival_insert / archival_search
```

**self-paging(MemGPT 的标志机制,复用 P3 的 context 压缩):**
- 活动上下文 ≥70% 预算 → 注入一条系统提示,诱导 agent 主动 `archival_insert`/`core_memory_replace` 落盘;
- ≥100% → 把中间回合递归摘要换页(`compact + llm_summarizer`),但**全量对话史仍在 ae_sessions**,agent 可用 `conversation_search` 翻回被换出的内容——这正是 recall 层存在的意义。

## 跑一遍(需 `ANVIL_DATABASE_URL` + `DEEPSEEK_API_KEY`)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
uv run anvil-ai-employee chat --memory letta
# 你:我叫小明,住在上海
# (它自己调 core_memory_replace / archival_insert 把信息写进记忆——你能在它的工具调用里看到)
# 退出再开
uv run anvil-ai-employee chat --memory letta
# 它的 system 里带着 core memory 块([human] 叫小明,住在上海),记得你
```

`--memory mem0` 切回 M2a 的托管式对照,`--memory none` 是无记忆基线。

## 一个微妙但重要的语义:读己写最终一致

agent 本轮调 `core_memory_replace` 改了 human 块,但本轮的 system 块(轮首已注入)**不会立刻刷新**——模型要到**下一轮** system_prefix 重新注入时才看到自己的编辑。这是 Letta 的真实行为(memory edits take effect at next context assembly),不是 bug。eval 因此**不假设同轮可见**,只在工具调用之后查库断言。

## 真实验证

- 全仓 `pytest -m "not live"` **643 绿**;Letta eval 断言 **agent 真·自调工具改库**(真 tool_use 往返,DB 实际被改)、self-paging flush 后 token 下降且无孤儿 tool 消息、archival 向量召回命中。
- **live 冒烟(真 deepseek-chat)**:对它说"我叫小明,住在上海",**真实模型自己调了记忆工具**把信息落库。`uv run pytest packages/ai-employee/tests/test_letta_eval.py -m live`(需 `DEEPSEEK_API_KEY`)。

## 复用与新建

- ✓ 真复用:M2a 的 `MemoryStrategy` 协议 / `MemoryVectorStore`(kinds=["archival"])/ `MemoryStore` / `SessionStore` / `chat.run_one_turn` 骨架;P3 `@tool`/`asyncbridge`/`context.{estimate_tokens,compact,llm_summarizer}`(self-paging);kb `FastEmbedEmbedder`;gateway/obs。
- ✗ 新建:`ae_core_blocks` + `CoreBlockStore`;`letta_tools.py`(5 工具);`LettaStrategy`;chat self-paging 接线;CLI `--memory letta`;Letta eval。

## 留待

M3(skills-as-markdown、Agent Inbox HITL)、M4(MCP 连接器)、M5(多员工编队)。至此 P4 三层记忆的**两种哲学(mem0 + Letta)都已实现并各有真模型验证**。
