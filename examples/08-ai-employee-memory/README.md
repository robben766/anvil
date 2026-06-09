# 08 — AI 员工·三层记忆(P4-M2a:mem0 哲学)

P4-M1 的员工只有一个"上次报到哪"的时间戳记忆。M2a 给它装上**托管式长期记忆(mem0 哲学)**:一个对话型助理,每轮对话后自动抽取关于你的事实、和已有记忆比对做增删改、存成向量;下轮/跨会话按向量召回,让它"记得你"。

> 经 5 人评审团评审后,原"双后端整套一个里程碑"拆为 **M2a(mem0,本篇)+ M2b(Letta 对照)**;skills-as-markdown 移到 M3。评审 8 条必修已并入实现。

## mem0 哲学:记忆由编排器管,agent 不知道记忆存在

```
你说一句
   │
   ▼
prepare_context ── embed_query(你的话) → 向量召回 top-k 已知事实 → 注入 system
   │                                                  ("# 关于用户你已知道:- 你住在上海")
   ▼
跑 P3 agent 循环出回复(agent 手里没有任何记忆工具)
   │
   ▼
after_turn ── LLM 抽取本轮关于你的离散事实
              └→ 逐事实找语义近邻的已有记忆
                 └→ LLM 比对决策 ADD / UPDATE / DELETE / NOOP   ← mem0 的心脏
                    └→ 写向量库(embed_texts,passage 侧)
```

对照将来的 M2b(Letta):**Letta 是 agent 自己调工具管记忆**(自管式)。M2a/M2b 并排就是为了吃透"谁来决定记什么/换页:编排器 vs agent"这个核心分野。

## 三层记忆里 M2a 做了哪两层

- **session(短期)**:`ae_sessions` 表持久化对话消息,可跨会话续聊。
- **longterm(长期·抽取事实)**:`ae_memories`(kind=fact,带 512d 向量)+ extract/reconcile 管线 + 向量召回。
- *skills(技能文件)*:M3 再做。

## 跑一遍(需 `ANVIL_DATABASE_URL` + `DEEPSEEK_API_KEY`)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
uv run anvil-ai-employee chat --memory mem0
# 你:我住在北京,养了一只猫
# 助理:好的,记住了。
# (退出,再开一次)
uv run anvil-ai-employee chat --memory mem0
# 你:我住哪来着?
# 助理:你住在北京。          ← 跨会话召回
# 你:我搬到上海了
# (它把"北京"那条记忆 UPDATE 成"上海",不是再加一条)
```

`--memory none` 切到无记忆基线做对照。

## 关键设计点(评审必修落实)

- **ADD/UPDATE/DELETE/NOOP 四操作**:漏掉 NOOP 会把"已存在的事实"误判成 ADD、记忆库重复膨胀——这是 mem0 论文的四操作,不能少。
- **eval 查记忆库、不查回复**:bot 能从对话上下文答对"上海"而库里静默存了双份。测试分两层断言:先证"召回到旧北京记忆",再证"reconcile 真把它 UPDATE"。
- **embed 方向铁律**:写库走 `embed_texts`(passage 无前缀),召回/找近邻走 `embed_query`(带检索前缀)——写反 cosine 召回质量塌,有单测锁死。
- **"只比语义近邻"是 mem0 固有软肋**:矛盾删除只在新旧事实互为向量近邻时才触发;eval 显式验证近邻假设成立,不假设。

## 真实验证

- 全仓 `pytest -m "not live"` 全绿;mem0 eval 的 5 类分层查库断言(召回/决策/NOOP/跨会话/embed 方向)全过。
- **live 冒烟(真 deepseek-chat + 真 bge 嵌入)**:跑"我住在北京"→"我搬到上海了",真实模型 reconcile 确实把居住地 UPDATE 成上海、未 double-ADD。`uv run pytest packages/ai-employee -m live`(需 `DEEPSEEK_API_KEY`)。

## 复用与新建(评审校正:把假复用改成真复用)

- ✓ 真复用:P3 harness(`run`/`@tool`)、`asyncbridge`、`FastEmbedEmbedder`(512d)、pgvector `Vector` 列、`guard.structured_chat`(抽取/比对强制 JSON)、gateway、obs。
- ✗ 假复用 → 新建:`PgVectorStore`(硬绑 ChunkRow)→ 新建 `MemoryVectorStore`(带 employee+kind 过滤);M1 的 `MemoryStore`(只 write/last)→ 扩 insert/knn/update/delete;`AgentState.new`(只 system+task)→ 加 `resume`/`from_messages` reducer(对话每轮一次性子运行)。

## 留待

M2b(Letta 自管式记忆 + self-paging + conversation_search + Letta eval)、M3(skills-as-markdown、Agent Inbox HITL)、M4(MCP)、M5(多员工)。
