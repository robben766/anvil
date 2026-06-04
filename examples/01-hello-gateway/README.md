# 01 · Hello Gateway

```bash
cp ../../.env.example ../../.env  # 填入你的 key
cd ../.. && uv sync --all-packages
set -a && source .env && set +a
uv run python examples/01-hello-gateway/main.py
```

连跑两次,观察第二次 `cached` 命中与成本差。

## 实测笔记(2026-06-04, deepseek-v4-flash)

| 固定前缀长度 | 第二次命中 |
|---|---|
| 120 token | 0(不命中) |
| 1330 token | **1280(96%)** |

旧文档称 DeepSeek 按 64-token 块做前缀缓存,实测 v4-flash 的生效门槛明显更高——
**短 prompt 永远不会命中**。想看到缓存效果,把 `SYSTEM` 扩到 1K token 以上再跑。
这正是"静态前缀放最前 + 测量而非想当然"两条缓存工程铁律的现场版。
