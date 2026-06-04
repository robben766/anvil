# 01 · Hello Gateway

```bash
cp ../../.env.example ../../.env  # 填入你的 key
cd ../.. && uv sync --all-packages
set -a && source .env && set +a
uv run python examples/01-hello-gateway/main.py
```

连跑两次,观察第二次 cached 命中(DeepSeek 默认开磁盘缓存)。
