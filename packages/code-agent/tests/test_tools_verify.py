import textwrap

from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.verify import run_tests


def _write(p, name, body):
    (p / name).write_text(textwrap.dedent(body))


def test_run_tests_passing(tmp_path):
    _write(tmp_path, "test_ok.py", "def test_ok():\n    assert 1 + 1 == 2\n")
    res = run_tests({"cmd": "python -m pytest -q"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "passed" in res.content.lower()


def test_run_tests_failing_reports_summary(tmp_path):
    _write(tmp_path, "test_bad.py", "def test_bad():\n    assert 1 + 1 == 3\n")
    res = run_tests({"cmd": "python -m pytest -q"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok is False
    assert "fail" in res.content.lower()


def test_run_tests_default_cmd(tmp_path):
    _write(tmp_path, "test_ok.py", "def test_ok():\n    assert True\n")
    res = run_tests({}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
