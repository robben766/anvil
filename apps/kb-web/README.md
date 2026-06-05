# kb-web

知识库产品前端 — 基于 RAG 的问答 UI,连接 kb-api(8400)通过 SSE 流式返回答案。

**栈:** Next.js 16 + React 19 + Tailwind 4(脚手架生成时落在 Next 16,lockfile 锁定,CI frozen-lockfile)

## 开发

```bash
pnpm install
pnpm dev      # http://localhost:3000
pnpm build    # 生产构建
pnpm lint     # ESLint 检查
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NEXT_PUBLIC_KB_API_URL` | `http://localhost:8400` | kb-api 地址;部署时指向真实后端 |

在项目根目录新建 `.env.local` 覆盖默认值:

```bash
NEXT_PUBLIC_KB_API_URL=http://localhost:8400
```
