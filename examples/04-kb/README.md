# example 04 — anvil-kb 知识库 E2E 演示

anvil-kb 是一个手工实现的 RAG 流水线（chunker / fastembed / PgVectorStore / Retriever / generate），
通过 FastAPI SSE 后端（kb-api）和 Next.js 前端（kb-web）提供产品级交互体验。
本目录记录了 KB-M1b 收官的 E2E 验收过程与生成侧评测结果。

---

## 三步起服务

### 1. 启动 anvil-postgres

```bash
cd /home/itachi/workspace/ai/anvil
docker compose -f infra/docker-compose.yml up -d anvil-postgres
```

### 2. 启动 kb-api（端口 8400）

```bash
set -a; source .env; set +a
uv run anvil-kb-api
```

`.env` 需包含：

```
ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
DEEPSEEK_API_KEY=<your-key>
```

### 3. 启动 kb-web（端口 3000）

```bash
cd apps/kb-web
pnpm install   # 首次
pnpm dev
```

---

## 上传语料

通过产品 API 路径上传（与前端同路径）：

```bash
for f in packages/kb/golden/corpus/*.md; do
  curl -s -F "file=@$f" http://localhost:8400/v1/kb/documents
  echo
done
```

三篇文档各返回 `201`，chunk_count 分别为 12 / 8 / 9。

---

## 提问与引用

### 步骤 1：打开 http://localhost:3000，文档面板列出 3 篇

![overview](screenshots/01-overview.png)

### 步骤 2：提问"等待期是多少天?"

回答含 90 天与 [1][2] 上标；展开检索来源条显示 5 条检索结果。

![answer with sources](screenshots/02-answer.png)

### 步骤 3：点击 [1] 上标 → 引用面板打开，黄色高亮等待期条款原文并自动滚动

![citation highlight](screenshots/03-citation-highlight.png)

### 步骤 4：提问拒答问题"这个产品覆盖牙科正畸吗?"

回答为"资料中未找到相关内容。"

![refusal](screenshots/04-refusal.png)

---

## 评测结果

### 检索侧（KB-M1a 首跑，`uv run anvil-kb eval`）

| metric | value |
| --- | --- |
| recall@5 | 1.0000 |
| precision@5 | 0.2000 |

### 生成侧（KB-M1b 首跑，`uv run python examples/04-kb/eval_generation.py`）

评测对象：`packages/kb/golden/kb.jsonl` 中 10 个 answerable 例。
Retriever k=5；judge 走网关 deepseek-chat；faithfulness + answer_relevancy。

| case | faithfulness | answer_relevancy |
| --- | --- | --- |
| kb-01 | 1.0000 | 0.7452 |
| kb-02 | 1.0000 | 0.8487 |
| kb-03 | 1.0000 | 0.8934 |
| kb-04 | 1.0000 | 0.7688 |
| kb-05 | 1.0000 | 0.8426 |
| kb-06 | 1.0000 | 0.7061 |
| kb-07 | 1.0000 | 0.8039 |
| kb-08 | 1.0000 | 0.7220 |
| kb-09 | 1.0000 | 0.7249 |
| kb-10 | 1.0000 | 0.8227 |
| **means** | **1.0000** | **0.7878** |

**overall**: 0.8939

- `faithfulness=1.0000`：生成答案的所有主张均可从检索内容中找到支撑，无幻觉。
- `answer_relevancy` 均值 0.7878：答案与问题相关性良好；kb-06（宽限期后果）最低 0.7061，原因是答案略侧重条款引用而非直接回答后果。
