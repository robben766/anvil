# KB-M1b:知识库产品壳(kb-api + kb-web)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 KB-M1a 的 RAG 库包成产品:FastAPI 薄壳(文档 CRUD + SSE 问答)+ Next.js 单页(上传/问答/引用面板高亮),live E2E 截图验收。

**Architecture:** spec §7/§8。apps/kb-api(uv workspace 成员,依赖 anvil-kb,薄壳无业务逻辑)+ apps/kb-web(Next.js 15 + pnpm,不进 uv workspace)。引用高亮用 chunk 的 start/end offset 切 `documents.content`,不做字符串再匹配。

**Tech Stack:** FastAPI / httpx ASGI 测试 / Next.js 15(App Router,TypeScript,Tailwind v4)/ pnpm 10 / node 24。kb-api 端口 **8400**;web 开发端口 3000,`NEXT_PUBLIC_KB_API_URL` 默认 `http://localhost:8400`。

**约定:** TDD(后端);commit `feat(kb-api)|feat(kb-web): ...`;镜像既有模式——auth/错误映射参照 `packages/core/gateway/src/anvil_gateway/proxy/app.py`,真 PG 测试参照 `packages/kb/tests/conftest.py`。前端无单测框架(刻意减法),以 `pnpm build` + ESLint + E2E 截图验收。

---

### Task B1: kb-api 骨架 + 文档 CRUD + 可选鉴权

**Files:** Create `apps/kb-api/pyproject.toml`(name=anvil-kb-api,module anvil_kb_api,deps: anvil-kb workspace、fastapi、python-multipart;scripts 入口 `anvil-kb-api = "anvil_kb_api.app:run"`)、`apps/kb-api/src/anvil_kb_api/app.py`、`apps/kb-api/tests/conftest.py`(复用 kb 测试库迁移逻辑 + ASGI client fixture)、`apps/kb-api/tests/test_documents.py`;Modify 根 `pyproject.toml`(members += "apps/kb-api";testpaths = ["packages", "apps"];known-first-party += "anvil_kb_api")。

端点(全部 `/v1/kb` 前缀):
- `POST /v1/kb/documents` multipart(field `file`,.md/.txt,UTF-8)→ `ingest_markdown`(title=文件名 stem,source_name=文件名)→ 201 `{id,title,source_name,chunk_count}`;空文档 ValueError → 400;非 .md/.txt → 400
- `GET /v1/kb/documents` → `[{id,title,source_name,chunk_count,created_at}]`(查 kb_documents + chunk 计数)
- `GET /v1/kb/documents/{id}` → 含 `content`(供高亮);404 处理
- `DELETE /v1/kb/documents/{id}` → 204(幂等)
- 鉴权:`ANVIL_KB_API_KEY` 设置时全端点要求 `Authorization: Bearer`,镜像 proxy/app.py 的实现与 401 语义
- app 构造:`create_app(session_factory=None, embedder=None) -> FastAPI`(None 时按 ANVIL_DATABASE_URL 懒构造;测试注入测试库 factory 与假 embedder);`run()` 用 uvicorn 起 8400

**Steps:** 失败测试(真 PG:upload→list→get(content 完整)→delete→404/400/401 路径;假 embedder one-hot)→ 实现 → `uv sync --all-packages && uv run ruff check . && uv run pytest -m "not live" -q` 全绿 → Commit。

### Task B2: POST /v1/kb/query(SSE + 非流式)

**Files:** Modify `apps/kb-api/src/anvil_kb_api/app.py`;Test `apps/kb-api/tests/test_query.py`。

请求 `{"question": str, "k": 5, "stream": true}`:
- `stream=false` → JSON `{"text", "citations":[{n,chunk_id,document_id,quote,header_path,start_offset,end_offset}]}`(调 `answer`)
- `stream=true` → SSE(`text/event-stream`),事件流:
  - `event: sources` `data: [{"n":1,"chunk_id","document_id","quote","header_path","start_offset","end_offset","score"}, ...]`(n=检索序 1-based)
  - `event: delta` `data: {"text": "..."}` ×N
  - `event: done` `data: {"text","citations":[...]}`
- chat 注入:`create_app(..., chat=None)`,透传到 generate 的 `answer/answer_stream`;测试注入 fake chat(非流式+流式两形态,镜像 KB-M1a test_generate 的 mock 写法)
- question 空白 → 400

**Steps:** 失败测试(SSE 用 httpx ASGI 流式读取断言三类事件顺序与 JSON 结构;citations 回链字段齐全)→ 实现(StreamingResponse;SSE 序列化器写成纯函数便于测)→ 全绿 → Commit。

### Task B3: kb-web 脚手架 + 文档管理页

**Files:** Create `apps/kb-web/`(`pnpm create next-app@15` 等价骨架:TypeScript、App Router、Tailwind v4、ESLint;**lockfile 必须提交**)、`src/lib/api.ts`(fetch 封装,base = `process.env.NEXT_PUBLIC_KB_API_URL ?? "http://localhost:8400"`)、`src/components/DocumentPanel.tsx`、页面 `src/app/page.tsx` 左右布局(左:文档面板;右:问答区占位)。

文档面板:上传(input file → POST multipart,显示 chunk_count)、列表(GET,含刷新)、删除(DELETE,确认后刷新)。错误显示为顶部红条。**不做**:多知识库、分页、用户体系。

**Steps:** 脚手架 → 实现 → `pnpm lint && pnpm build` 通过 → 手动 `curl` 确认 api 在 8400 时页面可上传列出(开发自测,真 E2E 在 B6)→ Commit(`.gitignore` 含 node_modules/.next)。

### Task B4: 问答 + SSE 渲染 + 引用面板高亮

**Files:** Create `src/components/ChatPanel.tsx`、`src/components/CitationPanel.tsx`、`src/lib/sse.ts`;Modify `page.tsx` 接线。

- `sse.ts`:`fetch(POST /v1/kb/query, {stream:true})` + ReadableStream 手写 SSE 解析(按 `\n\n` 分帧,解析 `event:`/`data:`;**不用 EventSource**——它不支持 POST)。回调 onSources/onDelta/onDone。
- ChatPanel:提问 → 先渲染 sources 折叠条(显示 k 条来源的 header_path+score)→ delta 流式追加答案文本 → done 后把答案中的 `[n]` 替换为可点击上标(正则 `\[(\d+)\]`,仅 citations 中存在的 n)。
- CitationPanel:点击 `[n]` → `GET /v1/kb/documents/{document_id}` → 渲染全文,`start_offset/end_offset` 切片三段式(前文淡色 / 命中段黄底高亮 / 后文淡色),滚动到高亮处;面板顶部显示 title + header_path。
- 加载/错误态齐全;流式中断(fetch abort)处理。

**Steps:** 实现 → `pnpm lint && pnpm build` → Commit。

### Task B5: CI node job + 文档

**Files:** Modify `.github/workflows/ci.yml`(新 job `web`:actions/setup-node@v4 node 24 + pnpm/action-setup@v4 → `pnpm install --frozen-lockfile` → `pnpm lint && pnpm build`,working-directory apps/kb-web;与 test job 并行)、`README.md`(KB 产品节:三条命令起全套——compose up anvil-postgres / `uv run anvil-kb-api` / `pnpm dev`;组件表状态 KB-M1b)、`CLAUDE.md`(apps 两条目:技术栈、端口、测试/构建命令)。

**Steps:** 改 → 本地等价验证(`pnpm install --frozen-lockfile && pnpm lint && pnpm build`)→ Commit → push 后看 CI 两 job 全绿。

### Task B6: live E2E 验收(截图)+ 生成侧 RAGAS live + 收尾

**Files:** Create `examples/04-kb/README.md`(walkthrough:起服务→API 上传 golden 语料→web 提问→引用高亮,嵌截图)、`examples/04-kb/screenshots/*.png`;Modify `packages/kb/golden/kb.jsonl`(reference 措辞统一:"本产品"→"本合同",KB-M1a 终审 Minor)。

**Steps:**
1. 起全套(anvil-postgres 已跑;`uv run anvil-kb-api` 后台;`pnpm dev` 后台),API 上传 golden 三篇
2. Playwright(浏览器工具)实测:打开 web → 提问"等待期是多少天?" → 等流式完成 → 点击 [1] → 验证引用面板高亮条款原文;截图 ≥3 张(整页/流式中/高亮态)存 examples/04-kb/screenshots/
3. 生成侧 live 评测:对 kb.jsonl 的 10 个 answerable 例,用 Retriever 检索 contexts + `answer` 生成,跑 anvil_eval 的 faithfulness/answer_relevancy(走真 key),记录首份生成侧报告到 examples/04-kb/README(分数如实,不达标不修饰)
4. 全量回归 + Commit + PR(标题 `P1 KB-M1b: kb-api + kb-web (upload/SSE chat/citation highlight)`)

---

## 自审记录
- spec §7 五端点全覆盖(B1/B2);§8 三块 UI(B3/B4)+ 检索调试视图明确留 KB-M2;§6 SSE 三事件协议字段对齐 KB-M1a 的 generate 实现
- 引用高亮数据流闭环:ChunkRow.offset(M1a)→ Citation(M1a)→ SSE done(B2)→ CitationPanel 切片(B4)→ GET content(B1)
- 占位扫描:无 TBD;端口/env 名/路径均显式
- 风险:Next.js 15 + Tailwind v4 脚手架版本漂移→B3 锁 lockfile;SSE in ASGI 测试→httpx stream 模式(KB-M1a 已有 async 迭代 mock 经验)
