# 11 — AI 员工·MCP 连接器(P4-M4:接外部世界)

M1-M3 让员工能定时干活、有三层记忆、高风险动作挂起等人审批。M4 让它**通过 MCP(Model Context Protocol)调用外部工具**(邮件/IM/日历/工单)——而且 **agent 永远拿不到长期凭证**,高风险 MCP 动作**自动**走 M3 的 Agent Inbox。

## 自研 client,吃透协议

不套 `mcp` PyPI SDK,手写一个 stdio JSON-RPC 2.0 client:把 MCP server 当子进程拉起,走 `initialize` 握手 → `notifications/initialized` 通知 → `tools/list` → `tools/call`。MCP stdio 帧 = **每行一个 JSON 对象**(不是 LSP 的 Content-Length)。

**会话生命周期是本期最大的坑**:MCP session 是长连接(initialize 一次,call 多次),而 `@tool` 是同步的。M1 的 `block_on`(每次新 event loop)绑不住子进程。解法:`McpClient` 持有一个**后台线程 + 独占 event loop**,子进程 transport 永远活在那个 loop 上,同步工具用 `run_coroutine_threadsafe(...).result()` 投递调用。

## 凭证服务端托管

`ae_mcp_tokens` 表按 (employee, connector, env_key) 存密钥;`McpClient` 在 spawn server 时把它们注入**子进程的 env**。agent 调 `gmail__send_email{to,subject,body}` 时参数里**没有 token**;结果文本里若回显了 token,client 自动脱敏成 `***`。

## 风险分级直接喂给 M3 HITL

读 MCP 工具的 annotation:`readOnlyHint`→low、`destructiveHint`(或无 hint)→high、其余 medium。`mcp_risk_policy` 把每个工具的真实 risk 喂给 M3 的 `hitl_run`:`calendar__list_events` 直接执行,`gmail__send_email` 挂起进 Inbox。**HITL/inbox/干预记忆全部不改**。

## 跑一遍(需 `ANVIL_DATABASE_URL`;run-mcp 需 `DEEPSEEK_API_KEY`)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil

# 1) 看连接器暴露了哪些工具 + 各自风险(起 mock server,握手,tools/list)
uv run anvil-ai-employee mcp list-tools --connector gmail
#   gmail__list_events  risk=low   hints=readOnlyHint
#   gmail__send_email   risk=high  hints=destructiveHint

# 2) 把凭证托管到服务端(agent 拿不到)
uv run anvil-ai-employee mcp put-token --connector gmail --env-key GMAIL_TOKEN --secret my-oauth-token --employee assistant

# 3) 读类任务:直接执行,不挂起
uv run anvil-ai-employee run-mcp --task "查 2026-06-09 的日程"

# 4) 写类任务:挂起进 Inbox;--auto-approve 在同进程内演示完整闭环
uv run anvil-ai-employee run-mcp --task "给 boss@x.com 发封周报邮件" --auto-approve
# → [run-mcp] 高风险 MCP 动作挂起进 Inbox。inbox_id=...
# → [run-mcp] --auto-approve:已执行 MCP 工具,结果(已脱敏):已发送 to=boss@x.com ... (via token=***)
```

**真实验证**:真子进程 mock server 完成 initialize 握手、tools/list 返回带 annotation 的两个工具;读类工具经 `hitl_step` 直接执行返回假日程;写类工具被挂起进 ae_inbox;approve 后 client 真发 JSON-RPC `tools/call`,server 读到 env 注入的 token 并回显,结果在 client 侧脱敏成 `***`。整条 spawn→handshake→list→call→HITL→脱敏在真子进程上走通。

## 跨进程 Inbox resume(本期边界)

M3 的 `inbox approve <id>` 用的是 demo registry(无 MCP 工具)。MCP 工具的 resume 需要在配置了连接器、client 活着的进程里做——所以 M4 的 demo 用同进程 `--auto-approve`。跨进程 MCP resume(worker 重建连接器)留作螺旋。

## 复用与新建

- ✓ 真复用:P3 `tools/base`(Tool/ToolRegistry)、`permission.risk_level`(policy fallback);M3 `hitl_run`/`hitl_step`/`apply_decision`/InboxStore/`resume_from_inbox`/干预写记忆(**全不改**);M1 worker/cli 骨架;M2a MemoryStore。
- ✗ 新建:`mcp/`(transport/client/tokens/adapter/connector + mock server);`ae_mcp_tokens` 表;CLI `mcp` 子命令 + `run-mcp`。

## 留待
真 OAuth 浏览器授权码流 / token refresh、SSE/HTTP 传输、MCP resources/prompts、凭证加密、跨进程 MCP resume、M5 多员工编队。
