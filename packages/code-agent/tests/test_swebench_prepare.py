import subprocess

from anvil_code_agent.eval.swebench import SweInstance, prepare_instance
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


def _origin(tmp_path):
    p = tmp_path / "origin"
    p.mkdir()
    (p / "m.py").write_text("def f():\n    return 1\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.t"],
                ["git", "config", "user.name", "t"], ["git", "add", "."],
                ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=p, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True
    ).stdout.strip()
    return p, sha


def test_prepare_instance_clones_applies_and_builds_task(tmp_path):
    origin, sha = _origin(tmp_path)
    # repo_url 用本地路径(file 源),绕开真实 GitHub
    inst = SweInstance(instance_id="m-1", repo=str(origin), base_commit=sha,
                       problem_statement="make f return 2", test_patch=TEST_PATCH,
                       fail_to_pass=["test_m.py::test_f"])
    dest = tmp_path / "work"
    task = prepare_instance(inst, str(dest), repo_url=str(origin))
    assert isinstance(task, Task)
    assert (dest / "m.py").is_file()
    assert (dest / "test_m.py").is_file()  # test_patch 已应用并提交
    assert "test_m.py::test_f" in task.verify_cmd
