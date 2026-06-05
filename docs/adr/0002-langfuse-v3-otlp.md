# ADR-0002: Upgrade Langfuse to v3 Stack for OTLP Ingestion

**Date:** 2026-06-05
**Status:** Accepted

---

## Context

M3 requires an OTLP HTTP endpoint to receive traces from the self-built `anvil-obs` exporter.
The existing deployment ran **Langfuse v2** (`langfuse/langfuse:2`) backed only by PostgreSQL.

Two blockers were evaluated:

1. **Missing OTLP endpoint** — Langfuse v2 has no `/api/public/otel` route.
   The OTLP ingestion API was introduced in **Langfuse v3.22+** and requires ClickHouse
   for event storage. There is no back-port and no workaround; v3 is the minimum viable version.

2. **Disk headroom** — An earlier assessment flagged a potential 12 GB constraint on the
   Docker data-root. Investigation confirmed that the Docker data-root is located on the
   `/home` partition (121 GB available). The constraint was a misread of the root-partition
   usage and does not apply. ClickHouse, MinIO, and the worker images fit comfortably.

The langfuse-managed PostgreSQL (`infra/data/pg`) holds only local-dev UI state (accounts,
project settings). The business ledger lives in `infra/data/anvil-pg` and is **never touched**
by this change. If the v2→v3 PG migration script fails, `infra/data/pg` can be wiped and
re-initialised without data loss.

---

## Decision

Upgrade the `infra/docker-compose.yml` Langfuse stack from v2 (1 service) to v3 (5 services):

| Service | Image | Purpose |
|---|---|---|
| `postgres` | `postgres:16-alpine` | Langfuse metadata (users, projects, API keys) — unchanged |
| `clickhouse` | `clickhouse/clickhouse-server:24.8-alpine` | Event/trace storage (required by v3) |
| `redis` | `redis:7-alpine` | Worker job queue |
| `minio` | `minio/minio:latest` | S3-compatible blob store for raw event payloads |
| `langfuse-worker` | `langfuse/langfuse-worker:3` | Async ingestion pipeline |
| `langfuse` | `langfuse/langfuse:3` | Web UI + public API (including OTLP endpoint) |

A YAML anchor (`&langfuse-env` / `*langfuse-env`) is used to share the full environment block
between `langfuse-worker` and `langfuse`, keeping the two in sync without duplication.

All credentials (NEXTAUTH_SECRET, SALT, ENCRYPTION_KEY, MinIO password, ClickHouse password)
are local-dev placeholder values. They must be rotated before any network-external exposure.

The `anvil-obs` exporter will POST protobuf-encoded `ExportTraceServiceRequest` messages to:

```
http://localhost:3100/api/public/otel/v1/traces
```

using HTTP Basic auth (`LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY`). These keys are obtained
from the Langfuse UI after first-run setup and stored in `.env` (not committed). Placeholder
lines are added to `.env.example`.

---

## Consequences

**Positive:**
- Standard OTLP protocol: any future backend (Grafana Tempo, Jaeger, Honeycomb) is a
  zero-code-change swap — only the endpoint env var changes.
- Langfuse v3 provides an LLM-specialized UI: token counts, cost, latency, and session
  grouping are first-class columns.
- The self-built `anvil-obs` exporter stays thin (only `opentelemetry-proto` for protobuf
  message classes; no opentelemetry-sdk dependency).

**Negative / Risks:**
- Five compose services instead of one; first-run startup takes 1–3 minutes for dual
  migrations (PostgreSQL schema + ClickHouse schema).
- MinIO, ClickHouse, and Redis add ~3–4 GB of Docker image layers on first pull.
- All new component credentials are local placeholder values — **must be changed** before
  any non-localhost exposure.
- If the v2→v3 PostgreSQL migration fails, recovery requires:
  ```bash
  docker compose -f infra/docker-compose.yml down
  rm -rf infra/data/pg infra/data/clickhouse
  docker compose -f infra/docker-compose.yml up -d
  ```
  (`infra/data/anvil-pg` must never be removed.)
