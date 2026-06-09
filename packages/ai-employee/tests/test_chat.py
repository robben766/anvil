import httpx
import pytest
import respx
from anvil_ai_employee.chat import run_one_turn
from anvil_ai_employee.memory.strategy import NoMemoryStrategy

pytestmark = pytest.mark.asyncio
DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _text(t):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": t},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


@respx.mock
async def test_run_one_turn_returns_reply_and_threads_history(session_factory):
    respx.post(DS_URL).mock(side_effect=[_text("你好呀"), _text("第二轮回复")])
    strat = NoMemoryStrategy()
    reply1, history1 = await run_one_turn(
        persona="你是助理",
        user_input="你好",
        history=(),
        strategy=strat,
        employee="assistant",
        session=None,
        model="deepseek-chat",
        max_steps=4,
    )
    assert reply1 == "你好呀"
    # history threads (user + assistant accumulated, minus system)
    reply2, history2 = await run_one_turn(
        persona="你是助理",
        user_input="再问",
        history=history1,
        strategy=strat,
        employee="assistant",
        session=None,
        model="deepseek-chat",
        max_steps=4,
    )
    assert reply2 == "第二轮回复"
    assert any(m["content"] == "你好" for m in history2)
    assert all(m["role"] != "system" for m in history2)


@respx.mock
async def test_run_one_turn_calls_strategy_hooks(session_factory, monkeypatch):
    respx.post(DS_URL).mock(return_value=_text("ok"))
    calls = {"prefix": 0, "after": 0}

    class SpyStrategy(NoMemoryStrategy):
        async def system_prefix(self, employee, user_msg):
            calls["prefix"] += 1
            return "记忆:你叫小明"

        async def after_turn(self, employee, session, msgs):
            calls["after"] += 1

    reply, _ = await run_one_turn(
        persona="助理",
        user_input="hi",
        history=(),
        strategy=SpyStrategy(),
        employee="assistant",
        session=None,
        model="deepseek-chat",
        max_steps=4,
    )
    assert calls["prefix"] == 1 and calls["after"] == 1
