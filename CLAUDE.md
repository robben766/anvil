# anvil — CLAUDE.md

## 常用命令
- 安装: `uv sync --all-packages`
- 测试: `uv run pytest -m "not live" -q`(live 冒烟: `uv run pytest -m live -q`,需 .env 配 key,手动跑)
- Lint: `uv run ruff check .`
- Langfuse: `docker compose -f infra/docker-compose.yml up -d` → http://localhost:3100

## 约定
- TDD:先写失败测试再实现;每个指标/解析逻辑必须有手算对照用例
- commit:conventional commits,英文 subject;作者邮箱用 GitHub noreply
- `.env` 永不入库;真实调用测试一律标 `@pytest.mark.live`
- 内容分级:本仓只放代码、脱敏设计文档、examples;文稿与内部笔记在私有笔记仓维护
- 设计文档:`docs/superpowers/specs/` 是唯一事实源,先读再改代码
