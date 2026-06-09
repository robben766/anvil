"""Golden multi-turn conversation fixture for the mem0 memory eval.

The script exercises the three behaviours the mem0 reconcile loop must get right:

1. **ADD** — the user states two fresh, stable facts early on: lives in 北京, has a 猫.
2. **UPDATE** — several turns later the user moves: 我搬到上海了. The居住地 fact must be
   *updated in place* (one row, now containing 上海), not double-ADDed (北京 + 上海 both
   alive). This is the mem0 "silent double-write" trap the eval must catch.
3. **NOOP** — the user repeats an already-known fact (我养了一只猫). The reconcile loop must
   recognise the equivalent neighbour and do nothing — facts row count stays flat.

The eval (tests/test_memory_eval.py) replays this script through ``Mem0Strategy`` one *user
turn* at a time, recording each turn's extract/reconcile ``structured_chat`` output via respx,
and then asserts **against ae_memories only** (never against the bot's reply text) across the
recall / decision / NOOP / cross-session / embed-direction layers.

``BEIJING_TO_SHANGHAI`` is a flat list of ``{"role", "content"}`` dicts (full transcript). The
eval slices it into per-turn user→assistant pairs; the indices below name the load-bearing turns.
"""

from __future__ import annotations

# Full transcript. Even indices are user messages, odd indices the assistant's replies, so the
# conversation decomposes into clean (user, assistant) pairs the eval feeds to ``after_turn``.
BEIJING_TO_SHANGHAI: list[dict] = [
    # turn 0 — ADD 居住地: 北京
    {"role": "user", "content": "你好,我住在北京。"},
    {"role": "assistant", "content": "你好!很高兴认识你,北京是个好地方。"},
    # turn 1 — ADD 宠物: 猫
    {"role": "user", "content": "对了,我养了一只猫。"},
    {"role": "assistant", "content": "好的,我记住了,你养了一只猫。"},
    # turn 2 — 闲聊,无可抽取事实(extract → 空)
    {"role": "user", "content": "今天天气怎么样?"},
    {"role": "assistant", "content": "我看不到实时天气,不过希望你那边是晴天。"},
    # turn 3 — 闲聊,无可抽取事实(extract → 空)
    {"role": "user", "content": "帮我推荐一部电影吧。"},
    {"role": "assistant", "content": "可以看看《让子弹飞》,挺经典的。"},
    # turn 4 — UPDATE 居住地: 北京 → 上海(矛盾轮,必须就地更新)
    {"role": "user", "content": "我搬到上海了。"},
    {"role": "assistant", "content": "好的,我记住了,你现在住在上海。"},
    # turn 5 — NOOP: 重复陈述同一事实(养猫),facts 行数不应增加
    {"role": "user", "content": "提醒你一下,我养了一只猫。"},
    {"role": "assistant", "content": "是的,我一直记得你养了一只猫。"},
]


def turns(transcript: list[dict] | None = None) -> list[list[dict]]:
    """Decompose the flat transcript into per-turn ``[user, assistant]`` pairs.

    Used by the eval to feed one turn at a time into ``Mem0Strategy.after_turn``."""
    script = BEIJING_TO_SHANGHAI if transcript is None else transcript
    return [script[i : i + 2] for i in range(0, len(script), 2)]
