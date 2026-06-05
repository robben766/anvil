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

通过产品 API 路径上传（与前端同路径），支持 .md、.txt、.pdf 格式：

```bash
# 上传 .md 语料（三篇）
for f in packages/kb/golden/corpus/*.md; do
  curl -s -F "file=@$f" http://localhost:8400/v1/kb/documents
  echo
done

# 或上传 .pdf 语料（PDF fixture，含页眉/页脚/表格版面陷阱）
curl -F "file=@packages/kb/golden/pdf/01-安康保障计划条款.pdf" http://localhost:8400/v1/kb/documents
```

三篇 .md 文档各返回 `201`，chunk_count 分别为 12 / 8 / 9；.pdf 返回 chunk_count=10（手工 PDF 解析器）。

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

---

## PDF 解析损耗实验（KB-M4）

将三篇 golden 语料的 PDF fixture（`packages/kb/golden/pdf/*.pdf`，含页眉/页脚/表格版面陷阱）替换 .md 直接灌入，在同一 golden 集（16 例，2 个拒答用例跳过，实际评分 14 例）上对比 hybrid 检索指标。

### 汇总对比

| corpus | recall@5 | precision@5 | MRR   | mean latency/query |
|--------|----------|-------------|-------|--------------------|
| .md 基线 | 1.000  | 0.200       | 0.881 | 131.2 ms           |
| PDF 管线 | 1.000  | 0.200       | 0.893 | 133.3 ms           |

**recall@5 两条管线均满分，PDF 管线 MRR 反而微升 +0.012（+1.4%）**，整体零损耗。

### 逐例对比（仅标注 MRR 有变化的用例）

| case  | 问题                       | .md MRR | PDF MRR | 变化    |
|-------|----------------------------|---------|---------|---------|
| kb-01 | 等待期是多少天?            | 1.000   | 0.500   | -0.500  |
| kb-07 | 犹豫期多少天?期内退保退多少? | 0.500 | 1.000   | +0.500  |
| kb-16 | 不想要这份保险了,刚买几天,钱能退吗? | 0.333 | 0.500 | +0.167 |
| 其余 11 例 | —                     | 均同    | 均同    | =       |

### 掉分例归因：kb-01（MRR 1.000 → 0.500）

**现象**：evidence `"本合同等待期为90天,自保险合同生效之日起计算。"` 仍在 top-5 中（recall=1.0），但从 #1 降至 #2。

**根因**：PDF 解析将 `03-产品说明.pdf` 里的等待期摘要行——`"本产品等待期90天,期内因疾病引起的出险不受保障。"`——切出为独立 chunk。该 chunk 与问题"等待期是多少天"的语义接近度在 hybrid RRF 排序中与条款原文（`01-安康保障计划条款.pdf` 第4条）打平，tie-breaking 碰巧将摘要 chunk 排在 #1。

注意：摘要 chunk 不含 evidence 的精确子串（"本合同等待期为90天"），因此 recall 依然计入条款 chunk，MRR 按 #2 计算为 0.5。这是一个**排序层面的 tie-break 问题**，而非解析丢字。

**md 中为何不出现**：.md 语料里 `01-安康保障计划条款.md` 的等待期 chunk 包含比产品说明更长、更具体的条款文本，dense 向量得分稍高，RRF 融合后保持在 #1。PDF 解析后两文档的等待期 chunk 措辞差距被拉近（均为一句话），导致得分接近。

### 提分例说明

- **kb-07 +0.500**：PDF 解析将犹豫期两条 evidence（"本合同设有15天犹豫期" + "犹豫期内投保人申请退保的,本公司全额退还已交保险费"）保留在同一 chunk 内，hybrid 排序将其提至 #1（md 中该 chunk 在 #2）。
- **kb-16 +0.167**：PDF 解析后犹豫期条款 chunk 语义更紧凑，与"刚买几天,钱能退吗"的 embedding 相似度微升，从 #3 升至 #2（MRR 0.333→0.500）。

### 解读

- **fixture 版面陷阱被解析器完全消化**：页眉（「星辉人寿」公司名）/ 页脚（页码）/ 多列表格（轻症给付比例表）全部正确剔除或转为 markdown，未发现因乱码、重复页眉、表格破坏而导致的 recall 下跌。
- **零损耗成立，MRR 略有提升**：净效果是 +1.4% MRR，原因是 PDF chunking 碰巧把若干 evidence 对齐得更好（kb-07/16 提升超过 kb-01 下跌）。
- **真实世界 PDF 会更难**：fixture 是用 reportlab 精心生成的规整 PDF，字体大小清晰分级、文字层完整；扫描件、双栏排版、手写批注、图片型 PDF 均不在当前 `parse_pdf` 的能力范围内，遇到这类情况损耗将显著。
- **排序 tie-break 脆弱性**：kb-01 掉分揭示了一个设计点——当两个 chunk 得分接近时，当前实现以 Python list 顺序（源自 SQL 返回行序）打破平局；如需稳定性，可在 RRF 层加入文档优先级或 chunk 位置因子。

### 截图：PDF 上传 + 引用面板

提问"轻症疾病最多能赔几次?"，点击 [1] 引用，面板展示 PDF 解析后的 markdown（含轻症给付比例表格）并高亮引用段落。文档列表中可见 `01-安康保障计划条款`（.pdf）条目。

![PDF citation panel](screenshots/07-pdf-citation.png)

上传命令（与 .md 相同路径，API 按 suffix 路由）：

```bash
curl -F "file=@packages/kb/golden/pdf/01-安康保障计划条款.pdf" http://localhost:8400/v1/kb/documents
# 返回 {"id":"...","title":"01-安康保障计划条款","source_name":"01-安康保障计划条款.pdf","chunk_count":10}
```
