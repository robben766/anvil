# packages/code-agent/tests/test_recovery.py
import json

from anvil_code_agent.harness.recovery import dump_state, load_state
from anvil_code_agent.state import AgentState


def test_dump_load_roundtrip():
    s = AgentState.new(system="s", task="t", workdir="/tmp/wt", max_steps=10)
    s = s.append({"role": "assistant", "content": "hi"}).advance()
    d = dump_state(s)
    # 必须 JSON 可序列化(断点落盘)
    blob = json.dumps(d)
    s2 = load_state(json.loads(blob))
    assert isinstance(s2, AgentState)
    assert s2.messages == s.messages
    assert s2.step == s.step
    assert s2.max_steps == s.max_steps
    assert s2.workdir == s.workdir
    assert s2.status == s.status


def test_loaded_state_is_resumable():
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=5)
    s = s.append({"role": "user", "content": "more"}).advance().advance()
    s2 = load_state(dump_state(s))
    # 还能继续推进(reducer 语义完好)
    s3 = s2.advance()
    assert s3.step == s.step + 1
    assert s3.status == "running"
