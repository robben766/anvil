"""Golden "agent self-calls memory tools" script for the Letta/MemGPT eval.

The contrast with the mem0 golden (``golden.py``) is the whole point: there the *orchestrator*
manages memory in ``after_turn`` (LLM extract → reconcile → write), and the agent never sees a
memory tool. Here the **agent itself** drives memory inside a single turn by emitting tool calls.

The eval (``tests/test_letta_eval.py``) replays this script through ``run_one_turn`` with a
``LettaStrategy`` and a respx ``side_effect`` list standing in for the model. The script:

1. The user states a fact ("我搬到上海了,顺便记一下我对花生过敏").
2. assistant turn 1 → ``core_memory_replace`` on the ``human`` block (北京 → 上海).
   (The eval seeds the ``human`` core block with 北京 first so there is an ``old`` to replace.)
3. assistant turn 2 → ``archival_insert`` storing a stable fact ("用户对花生过敏").
4. assistant turn 3 → plain text close ("已记下"), ``finish_reason="stop"`` ends the turn.

**Read-your-write is eventual (评审必修-8):** the ``core_memory_replace`` in step 2 writes the
DB, but ``messages[0]`` for *this* turn was assembled before the edit — the model does not see
its own edit until the *next* turn's ``system_prefix`` re-injects the core blocks. The eval
therefore asserts **only against the DB after ``run_one_turn`` returns**, never that the reply
text reflects the change.

``EXPECTED_TOOL_SEQUENCE`` names the tool calls the script drives, in order; the eval builds its
respx ``side_effect`` from it (one tool_calls response per entry, then a final text response).
Each entry is ``(tool_name, arguments_dict)``.
"""

from __future__ import annotations

# The user's single turn that triggers self-managed memory edits.
USER_INPUT = "我搬到上海了,顺便记一下我对花生过敏。"

# Seed for the ``human`` core block so step 2's replace has an ``old`` substring to hit.
HUMAN_BLOCK_SEED = "住在北京"

# The fact the agent files into archival long-term memory in step 3.
ARCHIVAL_FACT = "用户对花生过敏"

# Ordered tool calls the agent makes inside the turn. The eval turns each into a respx
# tool_calls response (in order), then appends a final plain-text "stop" response.
EXPECTED_TOOL_SEQUENCE: list[tuple[str, dict]] = [
    # 1) edit the always-resident human core block in place: 北京 → 上海
    ("core_memory_replace", {"label": "human", "old": "北京", "new": "上海"}),
    # 2) file a stable fact into vectorized archival memory
    ("archival_insert", {"text": ARCHIVAL_FACT}),
]

# Plain-text close that ends the turn (finish_reason="stop").
FINAL_REPLY = "好的,已记下:你现在住在上海,并且对花生过敏。"
