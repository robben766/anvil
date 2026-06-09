# 产品④ AI 员工 — M2b「自管式长期记忆(Letta/MemGPT 哲学)」设计

> 三层记忆的第二刀,M2a 的**纯增量对照**(MemoryStrategy 接口、ae_memories+embedding、MemoryVectorStore、chat REPL 已在 M2a 落地)。承接 5 人评审团对 Letta 维度的必修(recall 层 conversation_search 不可漏、self-paging 必做不可选、读己写最终一致)。

## 目标(一句话)

给对话型助理装上 **Letta/MemGPT 哲学的自管式记忆**:agent **自己调工具**管理记忆——常驻可自编辑的 core memory 块 + 向量化的 archival 外部记忆 + 可检索的 recall 对话史 + 上下文压力下的 self-paging 换页。与 M2a 的 mem0(编排器管记忆、agent 不知记忆存在)并排,吃透**"谁来决定记什么/换页:编排器 vs agent 自己"**这个核心分野。

## mem0 vs Letta 对照(本里程碑的教学卖点)

| | mem0(M2a,托管式) | Letta(M2b,自管式) |
|---|---|---|
| 谁管记忆 | 编排器(system_prefix 召回 + after_turn 抽取调和) | **agent 自己**(回合内调工具) |
| build_registry | 空(agent 无记忆工具) | **5 个记忆工具** |
| system_prefix | 向量召回事实注入 | 注入常驻 **core memory 块** |
| after_turn | LLM extract→reconcile→写库 | **no-op**(agent 已在回合内自己改) |
| 换页 | 不涉及(记忆是抽取的离散事实) | **self-paging**:上下文压力下 agent 被提示存记忆 + 递归摘要换页 |

## MemGPT 三层记忆(评审必修:recall 层不可漏)

| 层 | 存哪 | 谁可见/怎么访问 |
|---|---|---|
| **core memory** | 新表 `ae_core_blocks`(employee+label,如 persona/human) | 常驻 system 块,agent 用 `core_memory_append/replace` 自编辑 |
| **recall memory** | `ae_sessions.messages`(M2a 已持久化全量对话史) | agent 用 `conversation_search` 检索过去消息(self-paging 把活动上下文压缩后,全量史仍在此可查) |
| **archival memory** | `ae_memories` kind="archival"(带向量) | agent 用 `archival_insert/archival_search`(复用 M2a 的 MemoryVectorStore.knn kinds=["archival"]) |

## 架构

```
packages/ai-employee/src/anvil_ai_employee/
├─ db.py                  新增 ae_core_blocks 表
├─ memory/
│  ├─ coreblocks.py       新建 CoreBlockStore:get_all/append/replace(employee+label,字符上限)
│  ├─ letta_tools.py      新建 5 个 @tool(core_memory_append/replace, archival_insert/search, conversation_search)
│  └─ letta.py            新建 LettaStrategy:build_registry(返工具集)+ system_prefix(注 core 块)+ after_turn(no-op)
├─ chat.py                加 self-paging:活动上下文超阈值 → 注入警告系统消息 + compact 递归摘要(复用 M6)
└─ cli.py                 chat --memory 增 letta 选项
```

### 存储:新表 ae_core_blocks

```python
class CoreBlockRow(Base):
    __tablename__ = "ae_core_blocks"
    id: UUID pk
    employee: Text
    label: Text            # "persona" / "human"
    content: Text          # 块内文本(agent 可编辑)
    char_limit: Integer    # 默认 ~500;超限 append 返错信号
    updated_at: timestamptz
    # UniqueConstraint(employee, label)
```
archival 复用 ae_memories(kind="archival", embedding 非空);recall 复用 ae_sessions(无新表)。

### Letta 工具(`letta_tools.py`,P3 `@tool` 同步协议 + asyncbridge.block_on)

工具在 `build_registry(ctx)` 时闭包捕获一个 `LettaToolContext`(session_factory、embedder、employee、session_id)。**评审必修-8(读己写最终一致)**:`core_memory_replace` 写 DB 后,本轮 `messages[0]` 仍是旧 core 块(prepare 已注入),模型本轮看不到自己的编辑,下一轮 system_prefix 重注入才可见——这是 Letta 的真实行为(memory edits take effect next context assembly),文档写明、eval 不假设同轮可见。

- `core_memory_append(label, text)` — 往 core 块追加;超 char_limit 返回错误文本(ACI 信号,诱导 agent 转 archival),不写。
- `core_memory_replace(label, old, new)` — 块内子串替换;old 不在块内返错文本。
- `archival_insert(text)` — 存 archival(`embed_texts` 写向量,kind="archival")。
- `archival_search(query)` — `MemoryVectorStore.knn(employee, ["archival"], embed_query(query), k)`,返回命中文本。
- `conversation_search(query)` — 对 `ae_sessions.messages`(本 session)做关键词/子串检索,返回匹配的过去消息(recall 层;M2b 用确定性子串匹配,向量化留螺旋)。

工具失败永不抛异常崩循环(沿用 P3 铁律):返回可读失败文本当反馈。

### LettaStrategy(`letta.py`)

实现 M2a 已定的 `MemoryStrategy` 协议三钩子:
- `build_registry(ctx)` → `ToolRegistry([...5 个 letta 工具...])`(ctx 携带 employee+session_id+session_store,绑定工具)。
- `system_prefix(employee, user_msg)` → 读 `CoreBlockStore.get_all(employee)`,拼成 `"<core_memory>\n[persona] ...\n[human] ...\n</core_memory>\n你可以用 core_memory_*/archival_*/conversation_search 工具读写记忆。"`;无块时给默认空 persona/human 块(首次对话即初始化两块)。
- `after_turn(...)` → **no-op**(agent 已在回合内自管)。

**heartbeat / 工具链续跑映射(评审)**:Letta 的 request_heartbeat 让 agent 链式调多工具;P3 `run()` 的语义"模型继续调工具就继续、不调即 turn 结束"天然等价——文档写明此映射,无需额外机制。

### self-paging(评审必修:必做不可选,复用 code-agent M6)

在 chat 循环里(对 Letta 策略启用),活动上下文随多轮增长时:
- **warning(≥ warn_ratio×budget,默认 0.7)**:在本轮 msgs 里注入一条 system 提示消息——"上下文将满,请用 archival_insert/core_memory_replace 保存要点",诱导 agent 主动落盘。
- **flush(≥ budget)**:用 `compact(msgs, max_tokens=budget, summarizer=llm_summarizer(model))`(M6 已实现:把"中间回合"换成一条 LLM 递归摘要,cut 对齐非 tool 边界保配对)压缩活动上下文。**全量对话史仍在 ae_sessions 不丢**,agent 可用 `conversation_search` 找回被换出的内容——这正是 recall 层的意义。

实现:chat 加可选参 `paging`(token budget + warn_ratio + summarizer);Letta 路径开启,mem0/none 路径默认关(保 M2a 行为)。`estimate_tokens`/`compact`/`llm_summarizer` 全部从 `anvil_code_agent.harness.context` 复用。

### CLI

`anvil-ai-employee chat --memory letta`(M2a 已有 mem0|none,本期加 letta)。`make_strategy("letta", sf, model)` → `LettaStrategy(sf, embedder=FastEmbedEmbedder(), model=model)`。

## Letta eval(评审必修-8:断言 agent 自调工具,不假设同轮读己写可见)

`eval/memory/` 加 Letta 场景:respx 录一段 agent **自己调工具**的序列(assistant 调 core_memory_replace 改 human 块 → 调 archival_insert 存事实 → 调 archival_search 召回 → 文本收尾)。断言:
- **agent 真调了工具**:mocked side_effect 序列被消费,工具产生了 DB 变更——`CoreBlockStore.get_all` 里 human 块被改、`MemoryStore.list_facts(kind="archival")` 有新行。
- **不假设同轮可见**:断言只在工具调用"之后"查库,不断言"agent 本轮回复里已反映改动"。
- **archival 向量召回**:archival_search 命中刚存的事实。
- **self-paging**:构造一段超 budget 的历史,断言 flush 后活动 msgs token 数下降且保 tool_use 配对(无孤儿 tool 消息),全量史仍在 session。
- 单元用 respx + StubEmbedder(确定性);`@pytest.mark.live` 冒烟跑真 deepseek 让真 agent 自己用工具记一件事、下一轮召回。

## 范围边界(M2b 明确不做)

| 不做 | 留待 |
|---|---|
| recall 向量化(conversation_search 用子串匹配即可) | 螺旋回加 |
| core 块多于 persona/human 两个默认 label | 按需 |
| Agent Inbox / HITL 高风险挂起 | M3 |
| MCP 连接器 / 多员工编队 | M4 / M5 |
| skills-as-markdown(persona 文件化) | M3 |

## 复用与新建

- ✓ 真复用:M2a 的 MemoryStrategy 协议 / MemoryVectorStore(kinds=["archival"]) / MemoryStore(insert/list_facts) / SessionStore / chat.run_one_turn 骨架;P3 `@tool`/asyncbridge/`context.{estimate_tokens,compact,llm_summarizer}`(self-paging);kb FastEmbedEmbedder;gateway/obs。
- ✗ 新建:`ae_core_blocks` 表 + `CoreBlockStore`;`letta_tools.py` 5 工具;`LettaStrategy`;chat 的 self-paging 接线;CLI letta 选项;Letta eval。

## 错误处理

- 工具(core/archival/conversation)失败 → 返回可读失败文本当反馈,不崩循环。
- core_memory_append 超 char_limit / core_memory_replace 的 old 不存在 → 返回错误文本(ACI 信号),不写。
- self-paging flush 的 summarizer(LLM)失败 → 退回 Tier 1 仅截断(compact 本身已是这个降级行为),不崩对话。
- archival_search / conversation_search 空命中 → 返回"无匹配"。
