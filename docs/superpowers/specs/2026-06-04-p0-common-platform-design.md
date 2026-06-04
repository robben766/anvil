# P0 公共底座(anvil core)设计文档

> 状态:已获用户批准(2026-06-04 头脑风暴)。本文档是 P0 阶段的唯一事实源;实施计划据此产出。
> 上游依据:前期行业调研(框架全景 / 大师理念 / 通用参考架构,维护于私有笔记仓)。

## 1. 背景与目标

**项目**:anvil(铁砧)——AI 工程深度学习 monorepo。学习路线为"公共底座 + 四产品递进"(① 通用知识库 → ② 多模型协商会议 → ③ AI 团队编码 → ④ AI 员工)。

**P0 目标**:建成三件套薄版底座,后续所有产品的模型调用、可观测、评测都经过它:

1. **gateway** — 统一模型调用(DeepSeek + 阿里百炼 DashScope),含路由/fallback/成本记账(必含缓存命中测量)
2. **obs** — 自写 OTEL 标准 instrumentation,UI 用自托管 Langfuse
3. **eval** — RAGAS 四指标自实现 + golden set runner,可作 CI 门禁

**定位**:深度学习优先——核心逻辑自研造轮子吃透原理;框架(LiteLLM/Langfuse SDK/RAGAS 库)只做对照参考,不直接依赖其核心逻辑。

**非目标**(刻意减法):
- 不自建可观测 UI(用自托管 Langfuse 看图)
- 不做生产级网关特性(多租户、虚拟 key、配额管理)——产品需要时再加
- 不接入 DeepSeek/DashScope 以外的 provider(接口设计须可扩展,但不预先实现)
- 不做 L0 推理服务(vLLM)——无 GPU,原理学习靠精读笔记

**完成标准**:任意示例代码经 gateway 调通两家模型;每次调用在 Langfuse 可见完整 span(含 token/缓存命中/成本);`eval run` 能对一个 10 条的示例 golden set 跑出四指标分数并按阈值返回退出码。预期 2-3 周。

## 2. 已确认决策

| 决策 | 结论 | 理由 |
|---|---|---|
| 仓库形态 | Monorepo(uv workspace) | 原子化重构、一套 CI、IP 连载单一锚点 |
| 开源策略 | Day-1 公开 GitHub(账号 robben766),MIT | 倒逼 secrets 纪律;commit 即文章素材 |
| 仓库名 | `anvil` | 铁砧 = 锻造的底座;短、好记 |
| 架构形态 | 库优先 + 薄服务壳(方案 A) | 学习密度最高;一套核心体验 SDK/Proxy 双形态 |
| Provider | DeepSeek + DashScope | 现有可用 key;都 OpenAI 兼容;缓存 usage 字段不同,正好练适配 |
| 记账存储 | PostgreSQL + SQLAlchemy(async)+ Alembic | 2026-06-04 用户决策:不用 SQLite,直接生产形态;见 ADR-0001 |
| Langfuse | 自托管 v2(仅 Postgres) | v3 依赖 ClickHouse,较重;v2 够用 |
| Python | 3.12(uv 管理) | 不动系统 3.11 |
| 文章产出 | 每里程碑出公众号长文 + 小红书卡片 | 见 §8 |


## 3. 仓库骨架

```
anvil/
├─ packages/core/
│  ├─ gateway/        # anvil-gateway 包
│  ├─ obs/            # anvil-obs 包
│  └─ eval/           # anvil-eval 包
├─ apps/              # 产品①-④ 后续进驻(P0 为空)
├─ examples/          # 每篇文章一个可运行最小示例 examples/NN-主题/
├─ docs/
│  ├─ superpowers/specs/   # 设计文档(本文件)
│  └─ adr/                 # 架构决策记录
├─ infra/docker-compose.yml  # Langfuse v2 + Postgres
├─ pyproject.toml     # uv workspace 根
├─ .github/workflows/ci.yml  # ruff + pytest(live 测试除外)
├─ .env.example       # DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / LANGFUSE_*
├─ CLAUDE.md
├─ LICENSE            # MIT
└─ README.md          # 中英双语简介 + 学习路线图
```

**工程约定**:pytest + ruff + 强制类型标注;`.env` 永不入库;真实调用测试标记 `@pytest.mark.live`,CI 不跑;commit 遵循 conventional commits(英文 subject,正文可中文)。

## 4. gateway 包设计

### 4.1 核心接口

```python
# 统一入口(同步 + 流式)
async def chat(
    model: str,                    # "deepseek-chat" / "qwen-plus" / 别名
    messages: list[Message],
    *, stream: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[ToolSpec] | None = None,
    response_format: ResponseFormat | None = None,
    metadata: dict | None = None,  # 透传给记账/obs(如 session_id)
) -> ChatResponse | AsyncIterator[ChatChunk]
```

### 4.2 内部结构

```
gateway/
├─ client.py        # chat() 入口,组装 router+adapter+ledger
├─ adapters/
│  ├─ base.py       # ProviderAdapter 协议: send() / parse_usage() / classify_error()
│  ├─ deepseek.py   # usage: prompt_cache_hit_tokens / prompt_cache_miss_tokens
│  └─ dashscope.py  # usage: prompt_tokens_details.cached_tokens
├─ router.py        # 模型别名表 + fallback 链 + 重试(指数退避)
├─ usage.py         # UsageRecord 归一化模型(统一缓存字段语义)
├─ ledger.py        # 记账: PostgreSQL(SQLAlchemy async, Alembic 管 schema)
├─ errors.py        # 错误分类法(见 §4.4)
└─ proxy/app.py     # FastAPI 薄壳: POST /v1/chat/completions(OpenAI 兼容)
```

### 4.3 UsageRecord 归一化(本包的核心学习点)

```python
@dataclass
class UsageRecord:
    provider: str; model: str
    prompt_tokens: int; completion_tokens: int
    cached_tokens: int          # DeepSeek: prompt_cache_hit_tokens; DashScope: cached_tokens
    cache_hit_rate: float       # cached/prompt
    cost_cny: Decimal           # 按 provider 价表折算(缓存命中部分按折扣价)
    latency_ms: int; ttft_ms: int | None   # 流式时记首 token 延迟
    request_id: str; session_id: str | None; created_at: datetime
```

价表以代码内常量表维护(`pricing.py`),注明截止日期,变更走 ADR。

### 4.4 错误分类与 fallback 策略

| 类别 | 例 | 处理 |
|---|---|---|
| RETRYABLE | 429 限流、超时、5xx | 同 provider 指数退避重试 ≤2 次 → 仍失败则切 fallback 链下一家 |
| FATAL_REQUEST | 400 参数错、context 超长 | 直接抛出,不重试不切换 |
| FATAL_AUTH | 401/402(key 失效/欠费) | 抛出 + 标记该 provider 不可用(简单内存熔断,N 分钟后半开) |

fallback 链按"模型能力等价组"配置(如 `chat-default: [deepseek-chat, qwen-plus]`),不跨能力组降级。

## 5. obs 包设计

```
obs/
├─ trace.py         # @traced 装饰器 / span() context manager
├─ semconv.py       # gen_ai.* 字段常量(对齐 OTEL GenAI semantic conventions)
└─ exporter.py      # OTLP HTTP 导出 → Langfuse(批量、失败不阻塞主流程)
```

- gateway 在内部对每次 chat() 自动开 span,记录:`gen_ai.system`(provider)、`gen_ai.request.model`、token 三项、cached_tokens、cost、latency;prompt/completion 内容默认记录(学习项目,无隐私负担),提供开关。
- 上层应用可用 `@traced` 把任意函数挂为父 span,形成调用树。
- **导出失败只打日志,绝不影响主调用链路**(竖梁不能压垮主梁)。

## 6. eval 包设计

```
eval/
├─ dataset.py       # golden set JSONL: {id, question, reference, contexts?, tags}
├─ metrics/
│  ├─ faithfulness.py        # 拆主张→逐条比对 context(LLM-judge)
│  ├─ answer_relevancy.py    # 反推问题→embedding 余弦(用 DashScope embedding)
│  ├─ context_precision.py   # ranking-aware 精度(相关性默认由 judge 判定;golden set 带标注时用标注)
│  └─ context_recall.py      # reference 主张覆盖率(LLM-judge)
├─ judge.py         # LLM-as-judge 基建: rubric + 先理由后分 + 结构化输出 + 校准脚本
├─ runner.py        # 跑分 + 报告(markdown 表)+ 阈值退出码
└─ cli.py           # anvil-eval run --dataset xx.jsonl --threshold 0.8
```

- judge 默认用 deepseek-chat(便宜);**指标实现必须用手算样例做单元测试对照**(每个指标 ≥3 个手工标注用例,与 RAGAS 文档算例一致)。
- judge 调用本身也走 gateway → eval 的成本自动被记账(吃自己的狗粮)。

## 7. 数据流(一次调用的旅程)

```
应用代码 → gateway.chat()
  → router 解析别名/选 provider → adapter 发请求
  → 失败? errors.py 分类: RETRYABLE→退避重试→切链; FATAL→抛出
  → 成功: adapter.parse_usage() → UsageRecord 归一化
  → ledger 落 PostgreSQL(异步,不阻塞) + obs span 结束并导出
  → 返回 ChatResponse / 流式 chunk(usage 在最后一个 chunk)
Langfuse UI ← OTLP ← exporter(批量后台发送)
```

## 8. 配套内容产出

每完成一个里程碑(gateway / obs / eval 各为一个),产出:
1. 一篇深度技术文章(发布于微信公众号等渠道;文稿与配图在私有笔记仓撰写维护,本仓不存文稿)
2. 对应 `examples/NN-主题/` 可运行最小示例(读者 clone 即跑)

## 9. 测试策略

- **单元测试**:adapter 用 respx mock HTTP;router/errors/usage 纯逻辑直接测;eval 指标用手算样例对照
- **live 冒烟**:`@pytest.mark.live` 真实调两家 API(各 1 次,最小 token),本地手动跑,CI 跳过
- **CI**(GitHub Actions):ruff + 非 live pytest;P1 起加 eval gate
- ledger/client 测试使用真实 PostgreSQL(本地 compose / CI service container)

## 10. 里程碑

| # | 里程碑 | 验收 |
|---|---|---|
| M1 | 仓库初始化 + CI + compose 起 Langfuse | 骨架就绪,CI 绿 |
| M2 | gateway:双 adapter + router/fallback + 记账(含缓存测量) | 示例调通两家;PostgreSQL 有记录含 cache_hit_rate;文章① |
| M3 | obs:span 采集 + Langfuse 可见 | 调用树在 UI 完整呈现;文章② |
| M4 | eval:四指标 + runner + 手算对照测试 | 示例 golden set 出分;阈值退出码生效;文章③ |
| M5 | proxy 薄壳 + 收尾 | curl 走 OpenAI 格式调通;README 完整 |

## 11. 风险与对策

- **磁盘空间有限**:Langfuse 用 v2;docker 镜像定期清理
- **DashScope OpenAI 兼容层与原生接口差异**(如部分参数不支持):adapter 层吸收,差异记 ADR
- **价表过时**:pricing.py 注明核对日期;成本字段视为估算值
- **实现细节随调研对象版本漂移**(如 Langfuse v2 EOL):接口抽象保证可替换
