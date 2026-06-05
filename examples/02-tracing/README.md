# examples/02-tracing

演示自研 span + OTLP 导出到 Langfuse v3:一次业务调用形成完整调用树。

## 运行

```bash
# 在仓库根目录
set -a && source .env && set +a
uv run python examples/02-tracing/main.py
```

脚本会打印一句摘要文本到标准输出。

## 查看调用树

打开 http://localhost:3100 → Traces。

调用树结构:`task 模型网关`(根)→ `gateway.chat deepseek-chat`(两个)→ `chat deepseek-chat`(子 span),每个 `chat` span 携带属性:

- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
- `anvil.cached_tokens` / `anvil.cache_hit_rate`
- `anvil.cost_cny`
- `anvil.session_id`
