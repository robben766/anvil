"""Core memory blocks (MemGPT): small, always-in-context, agent-editable text blocks
keyed by (employee, label). Default labels: persona, human."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from anvil_ai_employee.db import CoreBlockRow

DEFAULT_LABELS = ("persona", "human")


class CoreBlockStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], *, char_limit: int = 500):
        self._sf = session_factory
        self._char_limit = char_limit

    async def get_all(self, *, employee: str) -> dict[str, str]:
        """Return {label: content}; lazily create empty persona/human blocks first time."""
        async with self._sf() as s:
            async with s.begin():
                rows = (
                    await s.execute(
                        select(CoreBlockRow).where(CoreBlockRow.employee == employee)
                    )
                ).scalars().all()
                have = {r.label for r in rows}
                for label in DEFAULT_LABELS:
                    if label not in have:
                        s.add(
                            CoreBlockRow(
                                employee=employee,
                                label=label,
                                content="",
                                char_limit=self._char_limit,
                            )
                        )
                rows = (
                    await s.execute(
                        select(CoreBlockRow).where(CoreBlockRow.employee == employee)
                    )
                ).scalars().all()
                return {r.label: r.content for r in rows}

    async def append(self, *, employee: str, label: str, text: str) -> bool:
        async with self._sf() as s:
            async with s.begin():
                row = (
                    await s.execute(
                        select(CoreBlockRow)
                        .where(CoreBlockRow.employee == employee)
                        .where(CoreBlockRow.label == label)
                    )
                ).scalar_one_or_none()
                if row is None:
                    row = CoreBlockRow(
                        employee=employee,
                        label=label,
                        content="",
                        char_limit=self._char_limit,
                    )
                    s.add(row)
                    await s.flush()
                new_content = row.content + ("\n" if row.content else "") + text
                if len(new_content) > row.char_limit:
                    return False
                row.content = new_content
                return True

    async def replace(self, *, employee: str, label: str, old: str, new: str) -> bool:
        async with self._sf() as s:
            async with s.begin():
                row = (
                    await s.execute(
                        select(CoreBlockRow)
                        .where(CoreBlockRow.employee == employee)
                        .where(CoreBlockRow.label == label)
                    )
                ).scalar_one_or_none()
                if row is None or old not in row.content:
                    return False
                candidate = row.content.replace(old, new)
                if len(candidate) > row.char_limit:
                    return False
                row.content = candidate
                return True
