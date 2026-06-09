"""Chat orchestration for a conversational AI employee.

Per-turn semantics (P3 ``harness.run`` is "run one task to done"): each chat turn is a
one-shot sub-run. We rebuild ``messages[0]`` = persona + memory prefix every turn, run to
done, take the last assistant message as the reply, let the strategy learn from the turn,
and accumulate the conversation into ``history`` (which never contains the system block —
that is rebuilt each turn from persona + fresh recall).
"""

from __future__ import annotations

import tempfile
from typing import Any

from anvil_code_agent.harness.loop import run
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext

from anvil_ai_employee.memory.strategy import MemoryStrategy


async def run_one_turn(
    *,
    persona: str,
    user_input: str,
    history: tuple[dict, ...],
    strategy: MemoryStrategy,
    employee: str,
    session: Any,
    model: str,
    max_steps: int,
) -> tuple[str, tuple[dict, ...]]:
    """One conversational turn as a one-shot sub-run.

    Returns ``(reply, new_history)`` where ``new_history`` is ``history`` with this turn's
    user + assistant messages appended (system block excluded — it is rebuilt each turn).
    ``session`` is only threaded through to ``strategy.after_turn``; persistence is the
    caller's job (see ``chat_repl``).
    """
    prefix = await strategy.system_prefix(employee, user_input)
    system = persona + ("\n\n" + prefix if prefix else "")
    user_msg = {"role": "user", "content": user_input}
    msgs = ({"role": "system", "content": system},) + tuple(history) + (user_msg,)

    with tempfile.TemporaryDirectory() as tmp:
        state = AgentState.from_messages(msgs, workdir=tmp, max_steps=max_steps)
        state = await run(
            state,
            model,
            strategy.build_registry(ctx=None),
            ToolContext(workdir=tmp),
        )

    reply = ""
    for m in reversed(state.messages):
        if m.get("role") == "assistant" and m.get("content"):
            reply = m["content"]
            break

    assistant_msg = {"role": "assistant", "content": reply}
    await strategy.after_turn(employee, session, [user_msg, assistant_msg])

    new_history = tuple(history) + (user_msg, assistant_msg)
    return reply, new_history


async def chat_repl(
    *,
    persona: str,
    strategy: MemoryStrategy,
    employee: str,
    model: str,
    session_store: Any = None,
    max_steps: int = 8,
) -> None:
    """Interactive REPL: read user lines, run a turn each, thread history, persist session.

    Exits on ``exit``, EOF, or KeyboardInterrupt. When ``session_store`` is given, a session
    is created up front and the threaded history (system-free) is saved after every turn.
    """
    sid = None
    if session_store is not None:
        sid = await session_store.create(employee=employee)
    session = (session_store, sid) if session_store is not None else None

    history: tuple[dict, ...] = ()
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip() == "exit":
            break
        reply, history = await run_one_turn(
            persona=persona,
            user_input=user_input,
            history=history,
            strategy=strategy,
            employee=employee,
            session=session,
            model=model,
            max_steps=max_steps,
        )
        print(reply)
        if session_store is not None and sid is not None:
            await session_store.save(sid, list(history))
