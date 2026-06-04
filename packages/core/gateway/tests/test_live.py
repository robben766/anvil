"""真实 API 冒烟:需要 .env 中的真实 key;手动运行 uv run pytest -m live -q"""

import pytest

from anvil_gateway import chat

pytestmark = pytest.mark.live


@pytest.mark.parametrize("model", ["deepseek-chat", "qwen-plus"])
async def test_live_minimal(model):
    resp = await chat(model, [{"role": "user", "content": "回复一个字:好"}], max_tokens=8)
    assert resp.content
    assert resp.usage.prompt_tokens > 0
