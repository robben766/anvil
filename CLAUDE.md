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
- M1 范围:单触发(cron)+ 单技能(周报员)+ 最小记忆;**M2 三层记忆(mem0+Letta)/ M3 Agent Inbox HITL / M4 MCP 连接器 / M5 多员工编队** 未做
