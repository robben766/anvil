from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.shell import bash


def test_bash_uses_executor_when_present(tmp_path):
    seen = {}

    def fake_executor(cmd):
        seen["cmd"] = cmd
        return (0, "from-executor")

    ctx = ToolContext(workdir=str(tmp_path), executor=fake_executor)
    res = bash({"cmd": "echo hi"}, ctx)
    assert res.ok
    assert res.content == "from-executor"
    assert seen["cmd"] == "echo hi"  # 路由到 executor,没走 host subprocess


def test_bash_executor_nonzero_is_error(tmp_path):
    ctx = ToolContext(workdir=str(tmp_path), executor=lambda c: (2, "boom"))
    res = bash({"cmd": "false"}, ctx)
    assert res.ok is False
    assert "exit code 2" in res.content
    assert "boom" in res.content


def test_bash_falls_back_to_host_without_executor(tmp_path):
    res = bash({"cmd": "echo host"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "host" in res.content
