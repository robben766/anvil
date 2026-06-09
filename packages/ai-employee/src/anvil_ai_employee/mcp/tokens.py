"""Server-side credential custody for MCP connectors. Secrets are stored per
(employee, connector, env_key) and handed to the MCP server subprocess as env vars at
spawn — the agent's tool-call arguments never carry them."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import McpTokenRow


class McpTokenStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._sf = session_factory

    async def put(self, *, employee: str, connector: str, env_key: str, secret: str) -> None:
        async with self._sf() as s:
            existing = (
                await s.execute(
                    select(McpTokenRow).where(
                        McpTokenRow.employee == employee,
                        McpTokenRow.connector == connector,
                        McpTokenRow.env_key == env_key,
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                existing.secret = secret
            else:
                s.add(
                    McpTokenRow(
                        employee=employee, connector=connector, env_key=env_key, secret=secret
                    )
                )
            await s.commit()

    async def env_for(self, *, employee: str, connector: str) -> dict[str, str]:
        async with self._sf() as s:
            rows = (
                (
                    await s.execute(
                        select(McpTokenRow).where(
                            McpTokenRow.employee == employee,
                            McpTokenRow.connector == connector,
                        )
                    )
                )
                .scalars()
                .all()
            )
            return {r.env_key: r.secret for r in rows}
