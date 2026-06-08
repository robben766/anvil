from anvil_code_agent.state import AgentState


def test_initial_state():
    s = AgentState.new(system="be a coder", task="fix bug", workdir="/tmp/wt", max_steps=10)
    assert s.step == 0
    assert s.status == "running"
    assert s.messages[0] == {"role": "system", "content": "be a coder"}
    assert s.messages[1] == {"role": "user", "content": "fix bug"}


def test_with_appended_is_immutable():
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=5)
    s2 = s.append({"role": "assistant", "content": "hi"}).advance()
    assert s.step == 0 and len(s.messages) == 2  # 原对象不变
    assert s2.step == 1 and len(s2.messages) == 3


def test_finish_sets_status():
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=5)
    assert s.finish("done").status == "done"
    assert s.finish("exhausted").status == "exhausted"
