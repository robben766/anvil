from anvil_code_agent.state import AgentState


def _state_done():
    s = AgentState.new(system="sys", task="hi", workdir="/tmp", max_steps=5)
    return s.append({"role": "assistant", "content": "done"}).advance().finish("done")


def test_resume_rearms_finished_state():
    s = _state_done()
    assert s.status == "done"
    r = s.resume({"role": "user", "content": "next"})
    assert r.status == "running"
    assert r.step == 0  # per-turn budget reset
    assert r.messages[-1] == {"role": "user", "content": "next"}
    assert len(r.messages) == len(s.messages) + 1


def test_from_messages_rehydrates():
    msgs = (
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    )
    s = AgentState.from_messages(msgs, workdir="/tmp", max_steps=7)
    assert s.messages == msgs
    assert s.status == "running" and s.step == 0 and s.max_steps == 7
