import shlex
import subprocess

import pytest
from anvil_code_agent.eval.swebench import (
    SweInstance,
    apply_test_patch,
    instance_to_task,
)
from anvil_code_agent.eval.task import Task

TEST_PATCH = (
    "diff --git a/test_m.py b/test_m.py\n"
    "new file mode 100644\n"
    "index 0000000..1111111\n"
    "--- /dev/null\n"
    "+++ b/test_m.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+from m import f\n"
    "+def test_f(): assert f() == 2\n"
)


def _repo(tmp_path):
    p = tmp_path / "r"
    p.mkdir()
    (p / "m.py").write_text("def f():\n    return 1\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.t"],
                ["git", "config", "user.name", "t"], ["git", "add", "."],
                ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=p, check=True)
    return p


def _inst():
    return SweInstance(instance_id="m-1", repo="x/m", base_commit="HEAD",
                       problem_statement="make f return 2", test_patch=TEST_PATCH,
                       fail_to_pass=["test_m.py::test_f"])


def test_apply_test_patch_commits_failing_test(tmp_path):
    p = _repo(tmp_path)
    apply_test_patch(str(p), _inst())
    # 测试文件已在 HEAD(被提交)
    assert (p / "test_m.py").is_file()
    tracked = subprocess.run(["git", "ls-files"], cwd=p, capture_output=True, text=True).stdout
    assert "test_m.py" in tracked


def test_apply_test_patch_bad_patch_raises(tmp_path):
    p = _repo(tmp_path)
    bad = SweInstance(instance_id="x", repo="x/m", base_commit="HEAD",
                      problem_statement="p", test_patch="not a real diff\n",
                      fail_to_pass=["t::t"])
    try:
        apply_test_patch(str(p), bad)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_instance_to_task_builds_failtopass_verify():
    t = instance_to_task(_inst(), "/some/repo")
    assert isinstance(t, Task)
    assert t.repo == "/some/repo"
    assert t.prompt == "make f return 2"
    assert "test_m.py::test_f" in t.verify_cmd
    assert "pytest" in t.verify_cmd


# FIX 1: empty FAIL_TO_PASS must raise ValueError
def test_instance_to_task_raises_on_empty_fail_to_pass():
    inst = SweInstance(
        instance_id="empty-1",
        repo="x/m",
        base_commit="HEAD",
        problem_statement="fix it",
        test_patch="",
        fail_to_pass=[],
    )
    with pytest.raises(ValueError, match="empty-1"):
        instance_to_task(inst, "/some/repo")


# FIX 2: node-ids with special chars must be shell-quoted in verify_cmd
def test_instance_to_task_quotes_nodeids_with_brackets():
    inst = SweInstance(
        instance_id="bracket-1",
        repo="x/m",
        base_commit="HEAD",
        problem_statement="fix it",
        test_patch="",
        fail_to_pass=["t.py::test_x[1-2]"],
    )
    t = instance_to_task(inst, "/some/repo")
    quoted = shlex.quote("t.py::test_x[1-2]")
    # The bracket must not appear bare (unquoted) in the command
    assert quoted in t.verify_cmd
    assert "t.py::test_x[1-2]" not in t.verify_cmd.replace(quoted, "")
