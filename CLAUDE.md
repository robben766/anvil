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
