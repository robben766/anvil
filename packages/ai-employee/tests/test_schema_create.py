import pytest
from anvil_ai_employee.db import JobRow
from sqlalchemy import select

pytestmark = pytest.mark.asyncio


async def test_tables_created_and_queryable(session_factory):
    async with session_factory() as s:
        rows = (await s.execute(select(JobRow))).scalars().all()
        assert rows == []
