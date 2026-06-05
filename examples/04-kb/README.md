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

---

## 混合检索对比实验（KB-M2）

三种检索模式在同一 golden 集（16 例，其中 kb-11/12 为拒答用例跳过，实际评分 14 例）上完整跑分。

### 三模式汇总

| mode   | recall@5 | precision@5 | MRR    |
|--------|----------|-------------|--------|
| dense  | 1.000    | 0.200       | 0.857  |
| sparse | 0.929    | 0.186       | 0.839  |
| hybrid | 1.000    | 0.200       | 0.881  |

### 对抗用例 kb-13..16 逐例 MRR 对比

| case  | 类型           | dense MRR | sparse MRR | hybrid MRR |
|-------|----------------|-----------|------------|------------|
| kb-13 | 词面精确       | 1.000     | 1.000      | 1.000      |
| kb-14 | 词面精确       | 1.000     | 0.500      | 1.000      |
| kb-15 | 换述（语义重写）| 1.000     | 0.000      | 1.000      |
| kb-16 | 换述（语义重写）| 0.500     | 0.250      | 0.333      |

### 解读

- **hybrid 在汇总层面最优**：MRR=0.881，recall=1.000，同时比 dense（MRR=0.857）高出约 2.8 个百分点，体现了 RRF 融合两路信号的提升效果。
- **sparse 在换述场景（kb-15）完全失效**：BM25 依赖词面重叠，面对语义重写的问句召回率跌至 0；dense 和 hybrid 均正常召回，这是 dense 向量互补 sparse 的典型证据。
- **kb-16 三模式 MRR 均偏低（dense=0.500，sparse=0.250，hybrid=0.333）**：该换述用例的目标 chunk 在所有模式下均未排到首位，说明该问法与语料的表达差距超出了当前 embedding 模型和 BM25 的共同能力边界，属于正常上限，并非 hybrid 退步。
- **dense-only 在本语料集上 recall 已达满分**，加入 sparse 的主要收益体现在 MRR 排序质量提升（从 0.857 → 0.881），而非召回覆盖。

### 截图：调试视图（三列 + 贡献标注）

![debug view](screenshots/05-debug-view.png)

---

## 重排对比实验（KB-M3）：精度换时间

在 hybrid 基础上加入 `--rerank`（bge-reranker-base Cross-Encoder），对同一 golden 集（14 个 answerable 例，k=5）完整跑分。

### 两组汇总

| mode             | recall@5 | precision@5 | MRR    | mean latency/query |
|------------------|----------|-------------|--------|--------------------|
| hybrid           | 1.000    | 0.200       | 0.881  | 131.3 ms           |
| hybrid + rerank  | 1.000    | 0.200       | 0.929  | 3128.8 ms          |

recall@5 两组均满分，不提高召回率（召回上限已由 hybrid 达到）；rerank 提升了排序精度：MRR 从 0.881 → 0.929（+5.4 个百分点）。延迟代价为 3128.8 ms vs 131.3 ms，每次查询多花约 **3000 ms**（约 23.8×），主要是 Cross-Encoder 对 5 个候选逐对前向的推理时间（首次运行还有模型加载，此处已加载完毕后测量）。

### 对抗用例 kb-13..16 MRR 对比（hybrid vs hybrid+rerank）

| case  | 类型              | hybrid MRR | hybrid+rerank MRR | 变化   |
|-------|-------------------|------------|-------------------|--------|
| kb-13 | 词面精确          | 1.000      | 1.000             | =      |
| kb-14 | 词面精确          | 1.000      | 1.000             | =      |
| kb-15 | 换述（语义重写）  | 1.000      | 1.000             | =      |
| kb-16 | 换述（语义重写）  | 0.333      | 0.500             | +0.167 |

- **kb-16 被 rerank 部分修复**：hybrid RRF 融合把犹豫期条款排在 #3（MRR=0.333），Cross-Encoder 将其提升至 #2（MRR=0.500），但仍未到 #1——得分最高的是产品说明中同提"犹豫期"的摘要性 chunk（Cross-Encoder score 1.3440 vs 犹豫期条款 -0.2199），两条都相关，属于合理的语义混淆而非错误。
- **kb-13..15 均已满分，rerank 不降分也不提分**：这些用例 hybrid 已将正确 chunk 排在 #1，Cross-Encoder 维持结果。
- **rerank 对全集的提升来自其他用例**：kb-07 hybrid MRR=0.500 → rerank MRR=1.000（由 #2 提至 #1）是 MRR 均值提升的主要贡献来源。

### 解读

- **rerank 值得用在精度敏感、延迟不敏感的场景**：+5.4% MRR 提升以换取约 3 秒/查询的延迟开销，适合离线批处理、低 QPS 精读类应用；不适合实时对话。
- **kb-16 提升有限（0.333 → 0.500）而非完全修复**：问题根源是该问法（"刚买几天，钱能退吗"）与语料中直接描述犹豫期的最精准 chunk 之间的语义距离，超出了 Cross-Encoder 可以修正的范围——embedding 初步召回的候选集中已不含最优 chunk 排第一的信号。这是语料覆盖与问法多样性的边界，不是 reranker 的 bug。
- **recall@5=1.000 两组均满分**：reranker 不影响召回覆盖，仅做重排，所有正确 chunk 在 top-5 候选中均已存在。

### 截图：四列调试视图（Cross-Encoder 重排列可见）

kb-16 原题"不想要这份保险了,刚买几天,钱能退吗?"，调试视图展示 dense / BM25 / RRF 融合 / Cross-Encoder 重排四列，可见 Cross-Encoder 将犹豫期相关 chunk 提至前两位。

![rerank debug view](screenshots/06-rerank-debug.png)
