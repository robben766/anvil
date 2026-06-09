# 产品④ AI 员工 — M4「MCP 连接器」设计

> 蓝图 §4.5 的"跨系统工具"落地。让 AI 员工通过**自研 MCP client**(JSON-RPC 2.0 over stdio)调用外部 MCP server 暴露的工具(邮件/IM/日历/工单)。三条铁律:① 自研 client+mock server,吃透 `initialize`→`tools/list`→`tools/call` 握手,不套 `mcp` PyPI SDK;② OAuth/凭证**服务端托管**,agent 的 tool args 永不含凭证;③ MCP 高风险动作**自动**走 M3 的 Agent Inbox 审批。复用 P3 的 `Tool`/`ToolRegistry`/`risk_level`、M3 的 HITL、M1 的 worker/CLI、M2 的干预记忆。

## 目标(一句话)

把任意 stdio MCP server 的工具注册成 P3 `@tool`,让 M1-M3 的 worker/HITL/memory 不加改造地驱动它们;凭证由服务端在 spawn 时注入 server 进程的 env,agent 全程只见工具名 + 业务参数。

## 核心机制一:自研 stdio JSON-RPC 2.0 传输

MCP 的 stdio 传输 = **newline-delimited JSON-RPC 2.0**(每条消息一行 JSON,消息体内不含裸换行;不是 LSP 的 `Content-Length` 帧)。client 把 MCP server 当子进程拉起,通过其 stdin 写请求、stdout 读响应,stderr 透传日志。

```python
# mcp/transport.py
class StdioTransport:
    """拥有一个 asyncio 子进程;收发 newline-delimited JSON-RPC 消息。"""
    def __init__(self, command: str, args: list[str], env: dict[str, str]): ...
    async def start(self) -> None:
        # asyncio.create_subprocess_exec(command, *args, env={**os.environ, **env},
        #   stdin=PIPE, stdout=PIPE, stderr=PIPE)
        ...
    async def send(self, msg: dict) -> None:
        # self._proc.stdin.write((json.dumps(msg) + "\n").encode()); await drain()
        ...
    async def receive(self) -> dict:
        # line = await self._proc.stdout.readline(); return json.loads(line)
        ...
    async def close(self) -> None:
        # 关 stdin、terminate、await wait(),超时则 kill
        ...
```

> 错误处理:`receive` 读到空行(EOF)→ server 已退出,抛 `McpTransportError`;`stderr` 内容在出错时附进异常文本,便于诊断 server 端崩溃。

## 核心机制二:McpClient 的会话生命周期(本期最大的坑)

MCP session 是**长连接**:`initialize` 握手一次,之后 `tools/call` 多次。但 ai-employee 的 `@tool` 函数是**同步**的,M1 用 `block_on`(每次新 `asyncio.run` 起一个一次性 loop)桥接异步 DB——这套**不能**用于 MCP,因为子进程 transport 绑定在某个 event loop 上,跨 `block_on` 调用会换 loop、连接失效。

**方案**:`McpClient` 持有一个**后台线程 + 该线程独占的 event loop**,子进程 transport 永远活在这个 loop 上。同步 `@tool` 通过 `asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)` 把调用投递进去并阻塞取结果。

```python
# mcp/client.py
PROTOCOL_VERSION = "2025-06-18"   # 我方支持版本;以 server initialize 返回的版本为准记录

@dataclass
class McpToolSpec:
    name: str                 # server 原始工具名(未加命名空间)
    description: str
    input_schema: dict        # JSON Schema(properties/required)
    annotations: dict         # {readOnlyHint?, destructiveHint?, idempotentHint?, ...}

class McpClient:
    """一个后台 loop 线程 + 一个 StdioTransport + 一个 MCP session。线程安全的同步 API。"""
    def __init__(self, *, connector: str, command: str, args: list[str], env: dict[str, str]): ...

    def start(self) -> list[McpToolSpec]:
        """启线程→loop→transport.start()→initialize 握手→initialized 通知→tools/list。
        返回工具清单。幂等:重复调用返回已缓存清单。"""

    def call_tool(self, name: str, arguments: dict, *, timeout: float = 60.0) -> str:
        """tools/call;把 MCP 结果的 content[] 拍平成纯文本返回(供 ToolResult.content)。
        协议级错误(isError 或 JSON-RPC error)→ 返回失败文本(ACI 铁律,不抛)。"""

    def close(self) -> None:
        """transport.close()→停 loop→join 线程。幂等。"""
```

**JSON-RPC 关联**:每个请求带自增 `id`;后台 loop 维护一个 reader task,把响应按 `id` 派发到等待的 `Future`;通知(无 `id`,如 server 端 log)忽略或透传。`initialize` 请求体:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":"2025-06-18",
  "capabilities":{},
  "clientInfo":{"name":"anvil-ai-employee","version":"0.1.0"}}}
```

握手后必须发 `{"jsonrpc":"2.0","method":"notifications/initialized"}` 通知,再 `tools/list`。

## 核心机制三:凭证服务端托管(agent 拿不到长期凭证)

```python
# mcp/tokens.py
class McpTokenRow(Base):
    __tablename__ = "ae_mcp_tokens"
    id: UUID pk
    employee: Text
    connector: Text           # 如 "gmail"
    env_key: Text             # 注入到 server 进程的环境变量名,如 "GMAIL_TOKEN"
    secret: Text              # 凭证值(本期明文存;加密留作螺旋)
    created_at: timestamptz
    # 唯一约束 (employee, connector, env_key)

class McpTokenStore:
    async def put(self, *, employee, connector, env_key, secret) -> None  # upsert
    async def env_for(self, *, employee, connector) -> dict[str, str]
        # 返回 {env_key: secret, ...},spawn server 时合进 env
```

**隔离保证**:`call_tool(name, arguments)` 的 `arguments` 来自 agent 的工具调用,**永不**包含 secret;secret 只在 `McpClient` 构造时由 `env_for` 注入子进程 env。脱敏:`call_tool` 返回前,把结果文本里出现的 secret 值替换成 `***`(防 server 回显凭证泄漏)。

## 核心机制四:MCP 工具 → P3 @tool 适配 + 风险映射

```python
# mcp/adapter.py
def mcp_risk(spec: McpToolSpec) -> str:
    a = spec.annotations or {}
    if a.get("readOnlyHint") is True:
        return "low"
    if a.get("destructiveHint") is True:
        return "high"
    if not a:                       # 无 annotation → 未知副作用 → 保守高危
        return "high"
    return "medium"

def mcp_tools(client: McpClient, specs: list[McpToolSpec]) -> list[Tool]:
    """每个 MCP 工具包成 P3 Tool:
      - 工具名 = f'{client.connector}__{spec.name}'(命名空间防撞车)
      - schema.params/required 取自 spec.input_schema
      - _fn(args, ctx) = ToolResult(client.call_tool(spec.name, args))(失败也返回文本)
    """
```

**风险接入 HITL**:`risk_level(name)` 当前是静态 dict、未知工具默认 `high`。MCP 工具名是 `gmail__send_email` 这类未知名,默认即 `high` → M3 的 `suspend_high` 自动挂起。但读类工具(`readOnlyHint`)应放行,所以需要让 HITL 策略能查到每个 MCP 工具的真实 risk。**做法**:新增 `mcp_risk_policy(specs)` —— 一个 `HitlPolicy`,按工具名查 `mcp_risk`(查不到的 fallback 到 P3 `risk_level`),传给 `hitl_run`。这样:

- `calendar__list_events`(readOnlyHint)→ low → 直接执行
- `gmail__send_email`(destructiveHint)→ high → 挂起进 Inbox

`apply_decision`/Inbox/干预写记忆**全部不改**——approve 后 `apply_decision` 调 `registry.dispatch('gmail__send_email', 原参, ctx)`,registry 里就是 MCP 适配的那个 Tool,于是真发 JSON-RPC。

## 核心机制五:Connector 配置组装

```python
# mcp/connector.py
@dataclass
class ConnectorConfig:
    name: str                 # "gmail"
    command: str              # "python"
    args: list[str]           # ["-m", "anvil_ai_employee.mcp.mock_servers.email"]
    # 凭证从 McpTokenStore 按 (employee, name) 取,不写在配置里

async def build_mcp_registry(
    *, configs: list[ConnectorConfig], employee: str, token_store: McpTokenStore
) -> tuple[ToolRegistry, list[McpClient], HitlPolicy]:
    """对每个 connector:env_for 取凭证 → McpClient(env=...).start() → mcp_tools。
    汇总所有 Tool 成 registry;汇总所有 spec 生成 mcp_risk_policy;返回 clients 供调用方 close。"""
```

## 与 worker / CLI 接线

- **worker**:跑 MCP 员工时,先 `build_mcp_registry` 起 clients,把 MCP tools 合进 registry,`hitl_run(..., policy=mcp_risk_policy)`;suspend → 落 ae_inbox(M3 不变);worker 退出前 `client.close()` 全部 connector。
- **CLI**(新 `mcp` 子命令组):
  - `anvil-ai-employee mcp list-tools --connector <name>` — 起 client、tools/list、打印工具名+risk+annotation 摘要,close。
  - `anvil-ai-employee mcp put-token --connector <name> --env-key <K> --secret <V>` — 写 ae_mcp_tokens。
  - `anvil-ai-employee run-mcp --persona <p> --task <t> --connector <name>` — demo:起 MCP registry 跑 HITL,读类直接执行、写类挂起进 Inbox(复用 M3 的 `inbox` 子命令处理后续 approve/resume)。
- **inbox resume 复用**:M3 的 `resume_from_inbox` 已是纯函数(load_state→apply_decision→record_intervention→hitl_run)。MCP 场景唯一差别是 registry 含 MCP tools 且 client 需活着——resume 时调用方负责 `build_mcp_registry` 再传入。`resume_from_inbox` 签名不变。

## Mock MCP server(真实验证用,手写 stdio server)

`mcp/mock_servers/email.py`:一个**零依赖**的 stdio JSON-RPC server,读 stdin 每行、按 method 派发:

- `initialize` → 回 `{protocolVersion, capabilities:{tools:{}}, serverInfo}`。
- `notifications/initialized` → 忽略(通知无响应)。
- `tools/list` → 回两个工具:
  - `list_events`:`annotations:{readOnlyHint:true}`,参数 `{date}`;返回假日程。
  - `send_email`:`annotations:{destructiveHint:true}`,参数 `{to,subject,body}`;**读 `GMAIL_TOKEN` env**,token 缺失则 `isError`,有则回 `已发送(via token=***)`(证明凭证经 env 注入、且结果脱敏)。
- 未知 method → JSON-RPC error `-32601`。

这个 mock 同时**反向验证**我们的 client 握手正确(它是按 spec 实现的最小 server)。

## 错误处理

- server spawn 失败 / 立即退出 → `start()` 抛 `McpTransportError`(含 stderr),worker 记 job 失败。
- `tools/call` 协议错误(JSON-RPC `error` 或结果 `isError:true`)→ `call_tool` 返回失败文本(ACI 铁律),agent 当反馈继续。
- `call_tool` 超时(`run_coroutine_threadsafe(...).result(timeout)` 超时)→ 返回超时文本,不杀 session。
- token 缺失 → server 端报错,经 `call_tool` 变成失败文本回给 agent(不在 client 侧硬失败,让 agent 看到"凭证未配置"反馈)。
- `close()` 幂等;重复 `start()` 返回缓存 specs。

## 范围边界(M4 明确不做)

| 不做 | 留待 |
|---|---|
| 真 OAuth 浏览器授权码流 / token refresh | 后续螺旋(TokenStore 存预置密钥即够证明隔离) |
| SSE / streamable-HTTP 传输 | 后续(stdio 已覆盖协议核心) |
| MCP resources / prompts / sampling | 后续(本期只做 tools) |
| 凭证加密存储(本期明文) | 后续 |
| 多 connector 自动发现 / 热插拔 | 后续(配置列举) |
| 多员工编队 | M5 |

## 复用与新建

- ✓ 真复用:P3 `tools/base.py`(`Tool`/`ToolRegistry`/`tool`/`ToolContext`/`ToolResult`)、`permission.risk_level`(MCP policy fallback);M3 `hitl.py`(`hitl_run`/`hitl_step`/`apply_decision`/`HitlPolicy`)、`inbox.py`(InboxStore)、`inbox_resume.py`(签名不变)、`hitl_memory.py`(干预写记忆);M1 worker/queue/cli 骨架、`ae_jobs`;M2a `MemoryStore`;obs span;`db.py` 的 Base/引擎。
- ✗ 新建:`mcp/` 子包(`transport.py`/`client.py`/`tokens.py`/`adapter.py`/`connector.py`/`mock_servers/email.py`);`ae_mcp_tokens` 表;`mcp_risk`/`mcp_risk_policy`;CLI `mcp` 子命令组 + `run-mcp`;example 11 README。

## 测试策略(TDD)

- `transport`:用一个 echo 子进程(`cat` 或小 python)测 send/receive newline 帧、EOF 抛错。
- `client`:**对真 mock server**(同包内 stdio server)测 `start()` 完成 initialize 握手 + tools/list 返回 2 工具;`call_tool('list_events')` 返回假日程;`call_tool('send_email')` 无 token→isError 文本、有 token→脱敏成功文本;`close()` 幂等。
- `tokens`:put/env_for round-trip(真 PG @5434),唯一约束 upsert。
- `adapter`:`mcp_risk` 三分支;`mcp_tools` 命名空间 + schema 映射 + `_fn` 调 client 返回 ToolResult;失败返回文本不抛。
- `mcp_risk_policy`:`list_events`→执行、`send_email`→挂起(对 `hitl_step` 端到端,真 mock server)。
- **集成 live(可选 marker)**:`build_mcp_registry`→`hitl_run` 真 deepseek 收"给 X 发邮件"→自调 `gmail__send_email`→挂起 Inbox→approve→`resume_from_inbox`→client 真发到 mock server→结果脱敏。
- 全程 `pytest -m "not live"` 必须绿(沿用 M1-M3 教训:每里程碑跑全仓收集防文件名/fixture 撞车)。
