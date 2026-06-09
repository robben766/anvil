# 产品④ AI 员工 — M2a「抽取式长期记忆(mem0 哲学)」设计

> 蓝图 §4.5 三层记忆的第一刀。**经 5 人评审团评审(GO_WITH_REVISIONS + RESCOPE)后定稿**:原"双后端整套一个里程碑"拆为 M2a(mem0 核心,先发)+ M2b(Letta 对照,后发);skills-as-markdown 砍出移到 M3。本 spec 仅覆盖 M2a。所有评审必修项已并入。

## 目标(一句话)

给 AI 员工装上"托管式长期记忆":一个对话型助理(CLI REPL)每轮对话后,由编排器用 LLM 抽取事实 → 与语义近邻的已有记忆比对做 ADD/UPDATE/DELETE/**NOOP** → 存向量;下轮/跨会话按向量召回注入,让它"记得你"。这是 mem0 哲学(编排器管记忆,agent 不知记忆存在),M2b 再做 Letta 的"agent 自管"对照。

## 范围边界(评审定)

| 本期(M2a) | 留待 |
|---|---|
| session 持久化(ae_sessions)+ MemoryStrategy 接口 + **Mem0Strategy** + chat REPL + mem0 eval | — |
| Letta 后端(agent 自管工具 + self-paging + conversation_search)+ Letta eval | **M2b** |
| skills-as-markdown(persona 文件化) | **M3** |
| 滚动会话摘要 S(抽取上下文的全局摘要段) | 螺旋回加(M2a 先只用"最近窗口") |

## 架构

```
packages/ai-employee/src/anvil_ai_employee/
├─ db.py                  扩 ae_memories(+embedding)、新增 ae_sessions
├─ memory/
│  ├─ store.py            (M1 有)MemoryStore 扩成完整 API:insert/knn/update/delete/last
│  ├─ vectorstore.py      新建 MemoryVectorStore.knn(employee, kinds, query_vec, k)(仿 PgVectorStore 写法)
│  ├─ strategy.py         MemoryStrategy 协议(三钩子)+ no-op 基类
│  └─ mem0.py             Mem0Strategy:system_prefix(召回注入)+ after_turn(extract→reconcile→apply)
├─ sessions.py            SessionStore:create/load/append_turn(ae_sessions 持久化,复用 recovery 序列化)
├─ chat.py                chat_repl + run_one_turn(每轮:prepare→resume→run→after_turn→存 session)
└─ cli.py                 (M1 有)加 `chat` 子命令

packages/code-agent/src/anvil_code_agent/state.py
└─ AgentState 加两个 reducer:resume(user_msg) / from_messages(...)（加法,不改既有行为）
```

### 评审必修-1:harness 适配(P3 `run()` 是"跑到底",对话要"每轮重置")

P3 `loop.run()` 在 `while status=='running'` 上循环,模型不调工具即 `finish('done')`;对已 `done` 的 state 再调 `run()` 是 no-op。`AgentState` 只有 `append/advance/finish`,**没有终态转回 running 的 reducer**。对话每轮需要:

给 `packages/code-agent/src/anvil_code_agent/state.py` 的 `AgentState` 加(纯加法):

```python
def resume(self, user_msg: Message) -> AgentState:
    """Re-arm a finished chat state for another user turn: append the user message,
    reset to running, and reset the per-turn step counter (max_steps is per-turn)."""
    return replace(
        self,
        messages=self.messages + (user_msg,),
        status="running",
        step=0,
    )

@classmethod
def from_messages(cls, messages: tuple[Message, ...], *, workdir: str, max_steps: int,
                  status: Status = "running") -> AgentState:
    """Rehydrate a state from a persisted message tuple (chat session resume)."""
    return cls(messages=messages, step=0, max_steps=max_steps, workdir=workdir, status=status)
```

**`max_steps` 是每轮预算**(resume 重置 step=0)——写进文档与测试,否则长会话永久 `exhausted`。

### 评审必修-2/-3:存储(假复用改新建)

**扩 `ae_memories`**(M1 已有 id/employee/kind/content/seq/created_at):
- 加 `embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)`(复用 `pgvector.sqlalchemy.Vector`,维度 512 与 kb 一致)
- `kind ∈ {fact, archival, core, report_marker}`。**不变量(app 层保证 + 注释)**:`fact`/`archival` ⇒ embedding NOT NULL;`core`/`report_marker` ⇒ NULL。
- 评审建议的 partial index `WHERE embedding IS NOT NULL` 标为可选优化(M2a 数据量小,可不加;加则在 conftest create 时一并)。

**新 `ae_sessions`**:
- `id` UUID pk / `employee` text / `messages` JSONB(对话消息列表,**不含每轮重注入的 system**,见下)/ `status` text / `created_at` / `updated_at` timestamptz

**`MemoryStore`(M1 只有 write/last)扩成完整 API**(near-total rewrite,仅复用 engine/session_factory):
```python
class MemoryStore:
    async def insert(self, *, employee, kind, content, embedding=None) -> uuid.UUID
    async def update(self, mem_id, *, content, embedding=None) -> None
    async def delete(self, mem_id) -> None
    async def last(self, *, employee, kind) -> str | None      # M1 兼容:report_marker
    async def list_facts(self, *, employee, kind="fact") -> list[MemoryRow]   # eval/调试用
```

**新 `MemoryVectorStore`**(仿 `PgVectorStore.search`,但带 employee+kind 过滤;**PgVectorStore 不可复用**——它硬绑 ChunkRow):
```python
class MemoryVectorStore:
    def __init__(self, session_factory): ...
    async def knn(self, *, employee: str, kinds: list[str], query_vec: list[float], k: int
                  ) -> list[tuple[MemoryRow, float]]:
        # SELECT * , embedding.cosine_distance(query_vec) AS distance
        # WHERE employee=? AND kind IN kinds AND embedding IS NOT NULL
        # ORDER BY distance LIMIT k  ; score = 1 - distance
```

### 评审必修-4/-5/补:Mem0Strategy

**MemoryStrategy 协议(M2a 落地,M2b 纯增量)** `memory/strategy.py`:
```python
class MemoryStrategy(Protocol):
    def build_registry(self, ctx) -> ToolRegistry: ...        # mem0: 基础工具(无记忆工具)
    async def system_prefix(self, employee: str, user_msg: str) -> str: ...  # 注入 messages[0] 的记忆文本
    async def after_turn(self, employee: str, session, msgs: list[dict]) -> None: ...  # 一轮后更新记忆
```
另给一个 `NoMemoryStrategy`(三钩子均空/返回空 registry)作基线与测试用,并让 no-op 半边显式命名。

**Mem0Strategy** `memory/mem0.py`(持有 db session_factory、embedder、model):
- `build_registry` → 基础(空)工具集——mem0 后端 agent 不持记忆工具(纯对话)。
- `system_prefix(employee, user_msg)`:
  1. `vec = embedder.embed_query(user_msg)`(**召回走 query 侧,带检索前缀**)
  2. `hits = MemoryVectorStore.knn(employee, kinds=["fact"], query_vec=vec, k=K)`
  3. 拼成"# 关于用户你已知道:\n- …"注入 system。无命中返回空串。
- `after_turn(employee, session, msgs)`:
  1. **抽取上下文(必修-4a)**:`window = 最近 N≈10 条 session 消息` + 当前 user/assistant pair。(滚动 summary S 标为螺旋回加,M2a 不做。)
  2. **extract**:`structured_chat(model, [...window...], schema={"required":["facts"]})`——prompt 含字面 "json"(json_object 要求),要求抽出关于用户的离散事实 list。
  3. 逐事实 `f`:`neighbors = MemoryVectorStore.knn(employee, ["fact"], embed_query(f), k=S≈5)`。
  4. **reconcile(必修-4b 含 NOOP)**:`structured_chat(model, reconcile_prompt(f, neighbors), schema={"required":["op"]})`,op ∈ `{ADD, UPDATE, DELETE, NOOP}`,UPDATE/DELETE 带 `target_id`。prompt 含字面 "json"。
  5. **Python 侧显式校验**(structured_chat 只查 required 存在性、不校验枚举值):`op` 非法 / `target_id` 不在 neighbors id 集 → 当 NOOP 兜底 + 记 obs。
  6. **apply**:ADD→`store.insert(kind="fact", content=f, embedding=embed_texts([f])[0])`(**写库走 passage 侧 `embed_texts`,无前缀**);UPDATE→`store.update(target_id, content, embed_texts)`;DELETE→`store.delete(target_id)`;NOOP→无操作。
  7. 整段包 `obs.span("ai_employee.mem0.after_turn", facts=len(...))`。
- **embed 方向铁律(必修-5/补)**:写库=`embed_texts`(passage 无前缀),召回/找近邻=`embed_query`(带中文检索前缀)。**写反 cosine 召回质量塌**——加单测锁死方向。
- 构造时 warm 一次 embedder。

### 会话与 chat 循环

**`SessionStore`** `sessions.py`:`create(employee)->id` / `load(id)->messages tuple` / `save(id, messages, status)`。持久化**只存对话消息**(messages 去掉每轮重注入的 system+memory 那条);复用 `recovery.dump_state` 的"messages 是普通 dict"事实直接存 JSONB。

**`run_one_turn`** `chat.py`(纯函数,可测):
```
1. prefix = await strategy.system_prefix(employee, user_input)
2. system = persona + ("\n\n" + prefix if prefix else "")     # 每轮重建 messages[0]
3. state = AgentState.from_messages((system_msg,) + history + (user_msg,), workdir=tmp, max_steps=M)
        # 或 history 为空时走 AgentState.new;已有 state 走 state.resume(user_msg) 并替换 messages[0]
4. state = await run(state, model, strategy.build_registry(ctx), toolctx)   # 跑到 done
5. reply = 末条 assistant text
6. await strategy.after_turn(employee, session, [user_msg, assistant_msg])
7. SessionStore.save(去掉 system 的 messages)
8. return reply, state
```
**`chat_repl`**:读 stdin → run_one_turn → print reply → 循环;`exit`/EOF 退出。

### 评审必修-6:mem0 eval(查记忆库,不查回复;召回/决策分层)

`eval/memory/`:golden 多轮对话 fixture(`住在北京` →(几轮后)`搬到上海` + 一条`重复陈述同一事实`)。断言**全部查 ae_memories**:
- **召回层**:抽出"上海"新事实后,`MemoryVectorStore.knn` top-k 近邻里**确实含旧"北京" fact 行**(隔离 embedder 召回,证明近邻假设成立——这是 mem0 固有软肋点)。
- **决策层**:矛盾轮后 `store.list_facts(employee)` 里"居住地"**恰一行 content 含"上海"**(UPDATE)或旧"北京"行 absent(DELETE)——断行数/内容 delta,**不是断回复文本**(bot 能从对话上下文答对"上海"而库里静默 double-ADD)。
- **NOOP**:重复陈述同一事实 → facts 行数不增。
- **跨会话**:新开 session,`system_prefix` 召回命中旧记忆。
- **embed 方向**:断言写库路径调 `embed_texts`、召回路径调 `embed_query`(用 spy/stub Embedder)。

**测试分层**:单元用 stub `Embedder`(Protocol 已支持,给确定性向量)+ respx 录 `structured_chat`(extract/reconcile)输出 → 确定性验证 reconcile 决策逻辑;`@pytest.mark.live` 冒烟跑真 deepseek + 真 FastEmbedEmbedder 全链(北京→上海真被 UPDATE)。chat 默认 `summarizer=None`(只截断),直到抽取被证能兜住对话事实。

## 复用清单(评审校正:假复用 → 新建)

- ✓ **真复用**:P3 `harness.loop`(step/run)、`tools/base`(@tool/ToolRegistry/ToolContext)、`asyncbridge.block_on`、`FastEmbedEmbedder`(512d)、`pgvector.Vector` 列类型、`guard.structured_chat`、`recovery`(messages 普通 dict)、gateway、obs、M1 的 queue/worker/CLI 骨架与 conftest。
- ✗ **假复用 → 新建(仿其写法)**:`PgVectorStore.search`(硬绑 ChunkRow)→ 新建 `MemoryVectorStore.knn`;`MemoryStore`(M1 仅 write/last)→ 扩成 insert/knn/update/delete;`AgentState.new`(仅 system+task)→ 加 `resume`/`from_messages` reducer。

## 落地风险排序(决定任务顺序)

1. **mem0 reconcile 正确性**(extract+近邻+UPDATE/DELETE 真改库不 double-ADD)—— 最高,novel core。
2. **session 多轮穿线**(run()-非-chat 失配,resume reducer)。
3. **MemoryVectorStore 向量查询**(employee+kind 过滤,PgVectorStore 非复用)。
4. skills-as-markdown —— 砍,移 M3。

## 错误处理

- 抽取/比对的 `structured_chat` 失败(StructuredOutputError)→ 该轮记忆更新跳过 + 记 obs,**不崩对话**(对话回复已产出)。
- reconcile 非法 op/target → NOOP 兜底。
- 向量召回空 → system_prefix 返回空串(首次对话无记忆是正常态)。
- embedder 加载失败 → 让其抛(配置问题,非运行态)。
