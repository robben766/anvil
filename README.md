# anvil 🔨

**Build your own AI stack, from scratch, to truly understand it.**

anvil(铁砧)是一个 AI 工程深度学习项目:自研造轮子实现 LLM 应用的核心模块,框架只做参照。底座之上将依次锻造四个产品:

```
packages/core(P0 公共底座)
  ├─ gateway   统一模型网关:多 provider / fallback / 成本与缓存命中记账
  ├─ obs       OTEL 标准可观测:自写采集层 + Langfuse 看图
  └─ eval      评测:手写 RAGAS 四指标 + golden set CI 门禁
apps/(递进式四产品)
  ① 通用知识库 → ② 多模型协商会议 → ③ AI 团队编码 → ④ AI 员工
```

设计文档:[docs/superpowers/specs/](docs/superpowers/specs/)

> Built in public. 每个里程碑配套一篇深度文章与可运行示例(examples/)。

## 快速开始

```bash
# 1. 安装依赖
uv sync --all-packages --all-extras

# 2. 配置环境变量
cp .env.example .env   # 然后填入真实 API key

# 3. 启动基础设施(Langfuse + PostgreSQL)
docker compose -f infra/docker-compose.yml up -d

# 4. 运行 gateway 示例
uv run python examples/01-hello-gateway/main.py

# 5. 运行 tracing 示例
uv run python examples/02-tracing/main.py

# 6. 运行评测
uv run anvil-eval run --dataset packages/core/eval/golden/demo.jsonl

# 7. 启动 OpenAI 兼容 proxy(见 examples/03-proxy/README.md)
uv run uvicorn anvil_gateway.proxy.app:app --port 8400

# 8. 运行全量测试(排除真实网络调用)
uv run pytest -m "not live" -q
```

## 组件

| 包 | 状态 | 一句话 |
|----|------|--------|
| anvil-gateway | M5 | 多 provider 统一调用 / fallback / 缓存计账 |
| anvil-obs | M3 | 自研 OTEL span + Langfuse 导出 |
| anvil-eval | M4 | 手写 RAGAS 四指标 + golden CI 门禁 |
| anvil-kb | KB-M1b | hand-rolled RAG:chunker/pgvector/retriever/generate;CLI ingest/query/eval |
| kb-web | KB-M1b | 知识库问答前端:Next.js 16 + SSE 流式答案 |

## 进度

- [x] M1 骨架 + CI + Langfuse
- [x] M2 gateway:统一调用 / fallback / 缓存命中记账 → [examples/01-hello-gateway](examples/01-hello-gateway/)
- [x] M3 obs:自研 span + OTLP 导出 Langfuse v3,GenAI semconv → [examples/02-tracing](examples/02-tracing/)
- [x] M4 eval:手写 RAGAS 四指标 + golden set CI 门禁 → `anvil-eval run --dataset packages/core/eval/golden/demo.jsonl`
- [x] M5 proxy:OpenAI 兼容 HTTP proxy shell(非流式 + SSE),curl 实测 DeepSeek → [examples/03-proxy](examples/03-proxy/)
- [x] KB-M1a 通用知识库核心:chunker / PgVectorStore / pipeline / Retriever / generate + CLI → `anvil-kb eval --dataset packages/kb/golden/kb.jsonl --corpus packages/kb/golden/corpus`
- [x] KB-M1b 知识库产品:FastAPI kb-api(SSE) + Next.js 前端 + CI web job

## 知识库产品(apps/)

三步启动全套知识库产品:

```bash
# 1. 启动 PostgreSQL(含 pgvector)
docker compose -f infra/docker-compose.yml up -d anvil-postgres

# 2. 启动知识库 API(端口 8400)
uv run anvil-kb-api

# 3. 启动前端(端口 3000)
cd apps/kb-web && pnpm install && pnpm dev
```

可选:先灌入演示语料(golden corpus),让问答有内容可检索:

```bash
uv run anvil-kb eval --dataset packages/kb/golden/kb.jsonl --corpus packages/kb/golden/corpus
```
