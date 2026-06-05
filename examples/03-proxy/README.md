# Example 03 — OpenAI-compatible Proxy

`anvil-gateway[proxy]` exposes the same `chat()` core as an HTTP service that
speaks the OpenAI wire protocol (`POST /v1/chat/completions`, non-stream + SSE).
Any client that works with OpenAI's API works here — no code changes, just point
`base_url` at the proxy.

## 启动

```bash
# 从仓库根目录执行
set -a && source .env && set +a
uv run uvicorn anvil_gateway.proxy.app:app --port 8400
```

服务起来后监听 `http://localhost:8400`。

## 非流式请求

```bash
curl -s http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat-default",
    "messages": [{"role": "user", "content": "用一句话解释什么是模型网关"}]
  }' | python3 -m json.tool
```

期望输出(节选):

```json
{
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens_details": {"cached_tokens": 0}}
}
```

## 流式请求 (SSE)

```bash
curl -sN http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat-default",
    "stream": true,
    "messages": [{"role": "user", "content": "数到3"}]
  }'
```

期望输出:每行一个 `data: {...}` SSE chunk，最后以 `data: [DONE]` 结束。

## 鉴权

默认不鉴权(本地学习场景)。设置环境变量后启用 Bearer 鉴权:

```bash
export ANVIL_PROXY_API_KEY=my-secret-key
```

之后所有请求需携带 `Authorization: Bearer my-secret-key`，否则返回 HTTP 401。

```bash
curl -s http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"model":"chat-default","messages":[{"role":"user","content":"hi"}]}'
```

## OpenAI Python SDK 直接使用

任何 OpenAI SDK 把 `base_url` 指过来即可用，无需改动其余代码:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8400/v1", api_key="anything")
response = client.chat.completions.create(
    model="chat-default",
    messages=[{"role": "user", "content": "Hello"}],
)
```

`api_key` 可填任意字符串(未设 `ANVIL_PROXY_API_KEY` 时不校验)。

## 错误映射

| 场景 | HTTP 状态码 |
|------|------------|
| 请求参数非法 (`FatalRequestError`) | 400 |
| API Key 无效 (`FatalAuthError`) | 401 |
| 所有 provider 均失败 (`AllProvidersFailedError`) | 502 |
