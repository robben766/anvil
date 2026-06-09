# anvil — CLAUDE.md

## 常用命令
- 安装: `uv sync --all-packages`
- 测试: `uv run pytest -m "not live" -q`(live 冒烟: `uv run pytest -m live -q`,需 .env 配 key,手动跑)
- Lint: `uv run ruff check .`
- Langfuse: `docker compose -f infra/docker-compose.yml up -d` → http://localhost:3100

## 约定
- TDD:先写失败测试再实现;每个指标/解析逻辑必须有手算对照用例
- commit:conventional commits,英文 subject;作者邮箱用 GitHub noreply
- `.env` 永不入库;真实调用测试一律标 `@pytest.mark.live`
- 内容分级:本仓只放代码、脱敏设计文档、examples;文稿与内部笔记在私有笔记仓维护
- 设计文档:`docs/superpowers/specs/` 是唯一事实源,先读再改代码

## anvil-kb (packages/kb)

hand-rolled RAG pipeline:chunker / fastembed / PgVectorStore(pgvector) / Retriever / generate(anvil_gateway chat-default)

- 测试: `uv run pytest packages/kb -q`(db/store 测试需真 PG(anvil_test 库),embed 测试需本地 fastembed 模型;其余走 mock)
- Golden 语料: `packages/kb/golden/corpus/*.md`(虚构保险产品 3 篇:条款/理赔指南/产品说明);PDF fixture: `packages/kb/golden/pdf/*.pdf`(同三篇,含页眉/页脚/表格版面陷阱);评测集: `packages/kb/golden/kb.jsonl`(50 例:换述/多跳/边界/拒答四类,含 evidences/answerable;防腐烂测试在 `packages/kb/tests/test_golden_dataset.py`)
- CLI 三命令(需 `ANVIL_DATABASE_URL` 环境变量):
  - `anvil-kb ingest <file.md|file.pdf ...>` — 写入 KB(.pdf 走 hand-rolled pdfplumber 解析器)
  - `anvil-kb query "<question>" [--k 5]` — 检索+生成(需 API key)
  - `anvil-kb eval --dataset kb.jsonl --corpus corpus/ [--k 5] [--recall-threshold 0.8] [--mode dense|sparse|hybrid] [--rerank] [--enrich]` — 纯检索评测,不调 LLM;exit 0=达标;hybrid 模式启用 pgvector + BM25 RRF 融合,是默认值;`--rerank` 加入 bge-reranker-base Cross-Encoder 精排(+MRR,+延迟 ~3 s/query);`--corpus` 同时支持 .md 和 .pdf 文件;`--enrich` 启用 Contextual Retrieval(灌入时每 chunk 调 LLM 生成 context_prefix,需 API key,prompt cache hit ~89%,25 chunk≈¥0.004)

## anvil-guard (packages/core/guard)

通用安全竖梁(圈1 普适):提示注入检测 + 结构化输出约束。

- `detect_injection(text) -> InjectionVerdict` — 确定性规则快路(中英双语,四类注入),纯函数零网络
- `detect_injection_llm(text)` — 可选 LLM 语义兜底(走 gateway,默认关闭)
- `structured_chat(model, messages, schema=...)` — 强制模型吐合法 JSON(json_object + 解析 + 重试一次),judge 已复用它
- 测试: `uv run pytest packages/core/guard -q`(走 respx mock,无需 key)
- 对抗集实验: `uv run python -m anvil_guard.experiments.injection_eval`
- 接线:kb CLI / kb-api 检索前拦截注入查询

## anvil-eval 校准(packages/core/eval)

- `anvil-eval calibrate --dataset golden/calibration.jsonl [--threshold 0.6]` — judge↔人工标注一致性(手写 Cohen's κ),低于阈值仅警告

## anvil-council (packages/core/council)

多模型评测陪审团(圈1 普适):多个不同模型独立按 rubric 打分 → 聚合 → 标分歧 → 校准。

- `score_case(case, models)` — 多陪审员并行独立打分(走 structured_chat)
- `aggregate(scores)` — 分项中位数 + 分歧探测 + 置信度
- `fleiss_kappa` / `compare_jury` — 多评委一致性 + 陪审团 vs 人工 vs 最佳单评委
- CLI: `anvil-council judge --dataset … --models deepseek-chat,qwen-plus` / `anvil-council calibrate --dataset …`
- 测试: `uv run pytest packages/core/council -q`(respx mock,无需 key);live 实验需 DEEPSEEK + 百炼 DASHSCOPE key
- 复用 gateway/guard/eval,编排原语供 P3 复用
- 实测(30 条):jury κ=0.626,qwen-plus 0.681,deepseek 0.526,评委间 Fleiss' κ=0.815 —— 评委高度冗余(非互补),弱评委稀释强评委,陪审团未跑赢最佳单评委(诚实负结果)

## apps/

### kb-api (apps/kb-api)

知识库 FastAPI 后端,端口 8400,SSE 流式返回答案。

- 启动: `uv run anvil-kb-api`(需 `ANVIL_DATABASE_URL` + API key)
- 测试: `uv run pytest apps/kb-api -q`(注意:api 测试需真 PG;mock 路径走 `not live`)
- DI 构造:依赖通过 FastAPI `Depends` 注入(retriever/generator 在 lifespan 初始化)

### kb-web (apps/kb-web)

知识库前端,Next.js 16 + React 19 + Tailwind 4,连接 kb-api SSE 端点。

- 安装: `pnpm install`
- 开发: `pnpm dev`(http://localhost:3000)
- 构建: `pnpm build`
- Lint: `pnpm lint`
- SSE 契约见 spec §6(请求格式) / §7(流式事件格式)

## anvil-code-agent (packages/code-agent)

自建编码 agent harness(圈3 agent):reducer 循环 + ACI 工具 + 闭环验证,worktree 隔离内修 bug。

- `harness/loop.py` — `step()` 一次 tool_use 往返;`run()` while 循环 + max_steps 守护
- `tools/` — read_file / edit_file(SEARCH-REPLACE+护栏)/ bash(超时截断)/ run_tests(闭环)
- `repo_map` / `grep`(M2)— tree-sitter+手写 PageRank 排序的符号地图 + 纯 Python regex 搜索,让 agent 自己定位代码
- `sandbox.py` — git worktree 隔离,可 diff 可丢弃
- `eval/` — Task + runner(worktree→agent→verify)+ pass 率;fixture: calc bug-fix
- CLI: `anvil-code-agent solve --repo <r> --prompt "<p>"` / `anvil-code-agent eval --dataset <tasks.jsonl>`
- 测试: `uv run pytest packages/code-agent -q`(工具/沙箱纯本地;loop/runner 走 respx mock + 测试 PG@5434);live 冒烟需 DEEPSEEK_API_KEY
- 复用 gateway(tool_use 往返)/obs(span 追踪每步工具);eval-pipeline 集成推迟到 M4(SWE-bench 基线)
- `harness/context.py`(M3)— 字符估 token + 结构安全压缩(只截老工具输出,不破坏 tool_use 配对),发送前按 token_budget 应用
- `harness/permission.py`(M3)— 工具风险分级(read/grep/repo_map=low,edit/run_tests=medium,bash=high,未知=high)+ 审批策略回调(auto_approve/deny_high/gate_by_risk),dispatch 前拦截,拦截转模型反馈
- `harness/recovery.py`(M3)— AgentState ↔ JSON dump/load,断点落盘可 resume
- `sandbox.py` DockerSandbox(M3)— docker CLI 起容器,workdir bind-mount 到 /work,`exec()` 容器内执行;`ToolContext.executor` 接缝让 bash 路由进容器(host 为默认回退)
- step/run 加 `policy`/`token_budget` 可选参,默认保持 M1 行为
- `eval/swebench.py`(M4)— SWE-bench(-Lite)实例适配器:SweInstance/load_instances(兼容 HF json-string 字段)/fetch_repo(clone@base_commit)/apply_test_patch(git apply 并提交,让 worktree HEAD 含失败测试)/instance_to_task(verify=跑 FAIL_TO_PASS)/prepare_instance(全链);**不重造官方 Docker harness**
- `eval/golden/baseline.jsonl`(M4)— calc/strops/counter 三个 bug-fix 任务(后两个多文件,逼 agent 用 repo_map/grep 定位)
- CLI: `anvil-code-agent eval --dataset .../baseline.jsonl`(本地基线 pass@1)/ `anvil-code-agent swebench --dataset <instances.jsonl> [--limit N]`(官方实例,clone 真实仓,live)
- `swebench --docker`(M5)— 每实例在 Docker 容器内隔离装依赖再跑:SweInstance 带 `image`(默认 python:3.11,含 gcc/git)+ `install_cmd`(如 `pip install -e .`);solve_task 的 `image`/`setup_cmd` 参在容器起来后先装仓依赖(失败即报错),再 agent 容器内跑 + 容器内 verify。把"依赖地狱"关进容器,是真跑官方 SWE-bench Lite 的前提(不造官方预构建镜像,临场装)
- `context.compact` 摘要 tier(M6)— 截断后仍超预算时,把"中间回合"整段替换成一条 LLM 摘要(`llm_summarizer(model)`,gateway 实现),替换边界对齐非 tool 消息保 tool_use 配对;step/run 加 `summarizer` 可选参(默认 None=只截断)
- `repo_map` 符号级排名(M6)— 每文件符号按全仓被引次数降序(枢纽符号优先)+ 每文件 top-K(`max_symbols_per_file`),比 M2 的文件级更细

## anvil-ai-employee (packages/ai-employee)

AI 员工(圈3 集大成,P4):cron 定时唤醒员工 → PG 队列 → worker 复用 P3 harness 跑 agent 循环 → 产出 + 写长期记忆。**M1「知识库周报员」** 是最小垂直骨架,串起前三个产品。

- `db.py` — 三表:`ae_schedules`(cron 计划)/ `ae_jobs`(PG 队列载体)/ `ae_memories`(最小长期记忆,M1 只存 `report_marker`,含单调 `seq` Identity 列保排序);复用 anvil_kb 的 `make_engine`/`make_session_factory`(一个 ANVIL_DATABASE_URL,一个 PG)
- `scheduler/queue.py` — PG 原生队列:`enqueue`/`claim_one`(`SELECT … FOR UPDATE SKIP LOCKED`,多 worker 不抢同一 job)/`complete`/`fail`,**无 Redis**
- `scheduler/trigger.py` — `Trigger` Protocol + `CronTrigger.due(now)`(croniter;插 job 与推进 `next_run_at` 同一事务原子提交);webhook/on-demand 触发可后续扩展不动 worker
- `memory/store.py` — `MemoryStore.write/last`(按 `seq` 降序取最新;M1 纯 recency 无向量,向量召回是 M2)
- `tools.py` — 周报员 ACI(P3 `@tool` 协议,**同步** fn):`recall_marker`/`kb_recent`(查 kb_documents)/`kb_search`(复用 P1 `Retriever` dense 模式)/`submit_report`(终止工具);工具内访问异步 DB/检索经 `asyncbridge.block_on`(ThreadPoolExecutor+asyncio.run,复用 code-agent M6 桥)
- `skills/kb_digest.py` — 技能 = persona(中文 system prompt 五步走)+ `build_registry(ctx)`
- `worker.py` — `run_once`:claim → 按 skill 取 registry → 跑 `anvil_code_agent.harness.run` → submit_report 已 complete job,worker 兜底 fail;整段包 obs span,单 job 异常隔离
- CLI: `anvil-ai-employee add-schedule --cron "0 9 * * 1" / tick [--loop] / work [--loop] / run-now / report --job <id>`
- 测试: `ANVIL_DATABASE_URL=...anvil_test uv run pytest packages/ai-employee -q`(queue/trigger/memory/tools/worker 需真 PG@5434;worker 用 respx mock gateway;conftest 的 engine fixture 同时建 ae 表 + kb 表,autouse `_gateway_env` 配 gateway);live 冒烟需 DEEPSEEK_API_KEY
- 复用:P3 harness(loop+@tool)、P1 知识库(Retriever/DocumentRow)、gateway(tool_use)、obs(span)
- M1 范围:单触发(cron)+ 单技能(周报员)+ 最小记忆

### M2a 抽取式长期记忆(mem0 哲学)

把 M1 的最小 report_marker 升级成三层记忆里的 session + longterm-fact 两层。**mem0 哲学=编排器管记忆,agent 不知记忆存在**(对照 M2b 的 Letta 自管式)。经 5 人评审团评审后拆出(原"双后端整套"→ M2a mem0 先发 / M2b Letta 后发;skills-as-markdown 移 M3),8 条必修已并入。

- `memory/strategy.py` — `MemoryStrategy` 协议(三钩子 build_registry/system_prefix/after_turn,mem0 与 Letta 共用)+ `NoMemoryStrategy` 基线
- `memory/mem0.py` — `Mem0Strategy`:`system_prefix`(embed_query 向量召回注入 system)+ `after_turn`(LLM extract → 逐事实找近邻 → structured_chat reconcile **ADD/UPDATE/DELETE/NOOP** → 写库)。**embed 方向铁律**:写库 embed_texts(passage)、召回/近邻 embed_query;reconcile op Python 侧校验非法当 NOOP;抽取/比对失败记 obs 不崩对话
- `memory/vectorstore.py` — `MemoryVectorStore.knn`(对 ae_memories 的 cosine 检索,**按 employee+kind 过滤**;PgVectorStore 硬绑 ChunkRow 不可复用,故新建)
- `memory/store.py` — M1 的 MemoryStore 扩 insert/update/delete/list_facts(保留 write/last)
- `sessions.py` — `SessionStore` 持久化对话消息(ae_sessions,跨会话续聊)
- `chat.py` — `run_one_turn`(每轮:system_prefix→from_messages→run→取末条 assistant→after_turn,history 不含 system)+ `chat_repl`
- `code-agent state.py` 加 `AgentState.resume/from_messages`(纯加法,对话每轮一次性子运行,max_steps 每轮重置)
- CLI: `anvil-ai-employee chat --memory mem0|none [--employee --model --persona]`
- eval: `eval/memory/golden.py`(北京→上海 fixture)+ 5 类**分层查库断言**(召回层/决策层/NOOP/跨会话/embed 方向,全查 ae_memories 不查回复)
- **真实验证**:live 冒烟(真 deepseek + 真 bge)跑北京→上海,真模型 reconcile 确实 UPDATE 不 double-ADD(`pytest packages/ai-employee -m live`,需 DEEPSEEK_API_KEY;conftest 已改成有真 key 就不塞 dummy)
- example: `examples/08-ai-employee-memory/`
- **留待:M2b(Letta 自管式 + self-paging + conversation_search + Letta eval)/ M3 skills+Agent Inbox HITL / M4 MCP / M5 多员工编队**

### M2b 自管式长期记忆(Letta/MemGPT 哲学)

M2a 的纯增量对照:**agent 自己调工具管记忆**(对照 mem0 编排器管)。完整实现 MemGPT 三层 core/recall/archival + self-paging。

- `db.py` 新增 `ae_core_blocks`(employee+label 唯一,常驻可编辑 core 块,字符上限)
- `memory/coreblocks.py` — `CoreBlockStore.get_all/append/replace`(惰性建 persona/human 默认块)
- `memory/letta_tools.py` — 5 个 `@tool`(agent 自调):`core_memory_append/replace`(core 层)、`archival_insert/archival_search`(archival 层,复用 MemoryVectorStore kinds=["archival"])、`conversation_search`(recall 层,对 ae_sessions.messages 子串检索);超限/old 不存在返 ACI 失败反馈不崩
- `memory/letta.py` — `LettaStrategy`:`build_registry` 返 5 工具、`system_prefix` 注 `<core_memory>` 块、`after_turn`=**no-op**(agent 回合内自管);**读己写最终一致**(本轮 core_memory_replace 下一轮 system 才可见,docstring 写明)
- `chat.py` — `apply_self_paging`(≥70% 注警告系统消息诱导落盘 / ≥100% 用 M6 `compact+llm_summarizer` 递归摘要换页;全量史仍在 ae_sessions 供 conversation_search 找回)+ run_one_turn 加 `paging` 可选参 + ctx 透传 employee/session_id(纯加法,mem0/none 忽略)
- CLI: `chat --memory letta`(make_strategy 加 letta 分支,letta 路径开 self-paging)
- eval: `eval/memory/letta_golden.py` + `test_letta_eval` respx 录 agent 自调工具序列,断言**查库变更(真 tool_use 往返改了 DB)+ 不假设同轮可见 + self-paging 无孤儿 tool**
- **真实验证**:live 冒烟真 deepseek **自己调记忆工具**记住"小明住上海"(`pytest packages/ai-employee/tests/test_letta_eval.py -m live`)
- heartbeat 映射:P3 run()"继续调工具就继续、不调即 turn 结束"天然等价 Letta request_heartbeat
- example: `examples/09-ai-employee-letta-memory/`
- **至此 P4 三层记忆两种哲学(mem0+Letta)都实现并各有真模型验证;留待 M3 skills+Agent Inbox HITL / M4 MCP / M5 多员工**

### M3 Agent Inbox(HITL 防跑飞)+ skills

蓝图§4.5 的"防跑飞"范式:高风险动作挂起等人审批,每次干预写回长期记忆(喂 M2)。

- **核心机制**:挂起点 = 最后一条 assistant 消息里尚无 tool 回复的 tool_call,AgentState 本身经 `recovery.dump_state` 完整承载;不改 P3 step()
- `hitl.py` — `HitlDecision`(EXECUTE/SUSPEND/DENY,StrEnum)+ `suspend_high` 默认策略 + `_unanswered_tool_calls` + `hitl_step`(每步只做一件事:处理一个待答工具高风险→`finish("suspended")`/否则执行,或调一次模型,advance 只在调模型时)+ `hitl_run` + `apply_decision`(approve 原参执行/edit 新参执行/reject 注入拒绝反馈不执行/respond 注入代答不执行,`replace(status="running")` 解挂)
- `db.py` 新增 `ae_inbox`(job_id/tool_name/tool_args/risk/state_json/status/decision/decision_payload)
- `inbox.py` — `InboxStore.suspend/list_pending/get/resolve`(resolve 幂等 where status=pending)
- `hitl_memory.py` — `record_intervention`(四动作各一句中文 → MemoryStore.insert kind="hitl" 带 embedding,可被 mem0 召回)
- `inbox_resume.py` — `resume_from_inbox`:load_state→记干预→apply_decision→hitl_run 续跑(闭环)
- `skills_loader.py` + `skills/*.md` — skills-as-markdown(三层记忆第三层,M2 砍出来的;persona 外置版本化 .md 运行时加载;hatchling 默认打包 .md 已验证)
- CLI: `inbox list/approve/edit/reject/respond` + `run-hitl` demo
- **真实验证**:真 deepseek 收"删日志"自己提出高风险 shell 调用→被挂起进 Inbox→reject 落 resolved+写干预记忆,整条挂起→审批→恢复在真模型走通;mock 测试覆盖四动作 + 闭环 suspend→resolve→resume→done
- example: `examples/10-ai-employee-hitl/`
- 留待:M4(MCP 令牌服务端托管)/ M5(多员工编队);Web Inbox UI / lease 超时 reclaim / 多级审批为后续

### M4「MCP 连接器」(examples/11)

- **自研 stdio JSON-RPC 2.0 client**(`mcp/transport.py` + `mcp/client.py`):不套 `mcp` SDK;`initialize`→`notifications/initialized`→`tools/list`→`tools/call`;stdio 帧 = 每行一个 JSON。
- **会话生命周期坑**:MCP session 是长连接而 `@tool` 同步;`McpClient` 用**后台线程+独占 event loop** 持有子进程 transport,同步工具走 `run_coroutine_threadsafe(...).result()`(不是 M1 的 block_on——那个每次换 loop 绑不住子进程)。
- **凭证服务端托管**(`mcp/tokens.py` + `ae_mcp_tokens` 表):密钥按 (employee, connector, env_key) 存,spawn server 时注入子进程 env;agent 的 tool args 永不含凭证;结果文本里的 token 在 client 侧脱敏成 `***`。
- **风险→HITL 零改造**(`mcp/adapter.py` + `mcp/connector.py`):`mcp_risk` 读 annotation(readOnlyHint→low / destructiveHint 或无 hint→high / 其余 medium);`mcp_risk_policy` 喂给 M3 `hitl_run`——读类直接执行、写类挂起进 ae_inbox;`apply_decision`/inbox/干预记忆全不改。
- 工具名命名空间 `{connector}__{tool}` 防撞车。mock server `mcp/mock_servers/email_server.py`(零依赖,reverse-validate 握手)。CLI:`mcp list-tools`/`mcp put-token`/`run-mcp [--auto-approve]`。
- 边界:跨进程 MCP inbox resume 需重建连接器(demo 用同进程 `--auto-approve`);真 OAuth/refresh、SSE/HTTP 传输、resources/prompts 留作螺旋。

### M5「多员工编队」(examples/12)

- **不新建编排引擎,直接复用 M1 PG 队列**:fleet 工作分发 = `ae_jobs` + SKIP LOCKED(M1 已证多 worker 无重复领取);job 加 `goal_id`(属哪个目标)+ `employee`(指派给谁),worker 按 `employee` 选 persona/registry。多开 `work --loop` = 并行编队。
- **supervisor**(`fleet/supervisor.py`):`decompose` 用 `guard.structured_chat` 把目标拆成独立子任务(非法/空→兜底单任务,绝不空转);`fan_out` 逐个 `enqueue`(带 goal_id+employee)。
- **aggregator**(`fleet/aggregator.py`):`children_terminal` 判全 done/failed;`aggregate` 综合各产出写 `ae_goals.result`,**失败子任务也纳入并标注缺失**(ACI 延伸到编队层),未全终态返 None、幂等。
- **team**(`fleet/team.py`):`EMPLOYEES` 注册表(kb_reporter 周报员 + researcher 调研员,角色异构);worker 泛化按 `job.employee` 选员工,`job.payload['task']` 作子任务(fallback 保 M1 行为)。
- 存储:`ae_goals` 表 + `JobRow.goal_id/employee`(nullable,M1-M4 单 job 路径不破)。CLI `team run --goal`/`team status --goal`。M2/M3/M4 员工天然可作编队成员。
- 边界:子任务 DAG 依赖、员工间消息/协商、动态扩缩容、陪审团择优综合留作螺旋。**至此 P4 与四产品主体全部完成。**

- 测试: `ANVIL_DATABASE_URL=...anvil_test uv run pytest packages/ai-employee -m "not live" -q`(全模块需真 PG@5434;respx mock gateway + StubEmbedder;**注意:live 与 mock 测试同跑会因真调用污染 respx 状态致 mock 失败——务必带 `-m "not live"`,CI 即此口径**;live 单独跑需 DEEPSEEK_API_KEY,conftest 有真 key 就不塞 dummy)
- 复用:P3 harness(loop+@tool+resume+context 压缩+permission 风险门+recovery 挂起恢复)、P1 知识库(Retriever/DocumentRow/FastEmbedEmbedder)、guard.structured_chat、gateway、obs
