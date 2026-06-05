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
- Golden 语料: `packages/kb/golden/corpus/*.md`(虚构保险产品 3 篇:条款/理赔指南/产品说明);评测集: `packages/kb/golden/kb.jsonl`(12 例,含 evidences/answerable)
- CLI 三命令(需 `ANVIL_DATABASE_URL` 环境变量):
  - `anvil-kb ingest <file.md ...>` — 写入 KB
  - `anvil-kb query "<question>" [--k 5]` — 检索+生成(需 API key)
  - `anvil-kb eval --dataset kb.jsonl --corpus corpus/ [--k 5] [--recall-threshold 0.8]` — 纯检索评测,不调 LLM;exit 0=达标
