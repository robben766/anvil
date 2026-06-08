from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.shell import bash


def test_bash_runs_and_captures_stdout(tmp_path):
    res = bash({"cmd": "echo hello"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "hello" in res.content


def test_bash_runs_in_workdir(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    res = bash({"cmd": "ls"}, ToolContext(workdir=str(tmp_path)))
    assert "marker.txt" in res.content


def test_bash_nonzero_exit_is_not_ok_but_readable(tmp_path):
    res = bash({"cmd": "exit 3"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok is False
    assert "exit code 3" in res.content


def test_bash_timeout(tmp_path):
    res = bash({"cmd": "sleep 5"}, ToolContext(workdir=str(tmp_path), timeout=0.3))
    assert res.ok is False
    assert "timed out" in res.content.lower()


def test_bash_truncates_long_output(tmp_path):
    res = bash(
        {"cmd": "for i in $(seq 1 5000); do echo line$i; done"},
        ToolContext(workdir=str(tmp_path), max_output=300),
    )
    assert res.truncated
    assert "truncated" in res.content.lower()
