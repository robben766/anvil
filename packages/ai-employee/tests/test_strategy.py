import pytest
from anvil_ai_employee.memory.strategy import NoMemoryStrategy

pytestmark = pytest.mark.asyncio


async def test_no_memory_strategy_is_noop():
    strat = NoMemoryStrategy()
    reg = strat.build_registry(ctx=None)
    assert reg.schemas() == []
    assert await strat.system_prefix("u1", "hi") == ""
    # after_turn must be awaitable no-op
    await strat.after_turn("u1", None, [])
