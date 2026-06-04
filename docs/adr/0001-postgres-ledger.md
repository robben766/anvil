# ADR-0001: 记账存储从 SQLite 迁移至 PostgreSQL

**Date**: 2026-06-04
**Status**: Accepted

## Context

P0 设计文档初版将记账存储定为 SQLite 单文件起步(零运维,经 repository 接口抽象可换 Postgres)。

## Decision

用户明确要求直接采用生产形态,不用 SQLite。决策:

- 记账层采用 **PostgreSQL + SQLAlchemy(asyncio)+ Alembic**
- ORM 模型定义在 `db.py`,schema 变更一律走 Alembic migration
- 使用 `NullPool` 简化连接池生命周期(低 QPS 可接受;高 QPS 时可换连接池)
- 测试层同样使用真实 PostgreSQL:本地通过 `infra/docker-compose.yml` 的 `anvil-postgres` 服务(端口 5434),CI 通过 GitHub Actions service container

## Consequences

- 本地开发需先启动 `anvil-postgres`:  
  `docker compose -f infra/docker-compose.yml up -d anvil-postgres`
- 首次或 schema 变更后需运行迁移:  
  `cd packages/core/gateway && uv run alembic upgrade head`
- CI 需要 service container(已在 `.github/workflows/ci.yml` 配置)
- 为产品①(通用知识库)的多表 schema 演进铺路,Alembic 机制已就位
- 彻底移除 SQLite 依赖:代码、配置、`.gitignore` 均无 sqlite 残留
