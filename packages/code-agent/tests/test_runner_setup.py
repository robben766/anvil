import inspect
import subprocess

import pytest
from anvil_code_agent.eval.runner import solve_task
from anvil_code_agent.eval.task import Task


def test_solve_task_signature_has_image_and_setup_cmd():
    sig = inspect.signature(solve_task)
    assert "image" in sig.parameters
    assert "setup_cmd" in sig.parameters
    assert sig.parameters["image"].default is None
    assert sig.parameters["setup_cmd"].default is None


async def test_setup_cmd_failure_raises(tmp_path, monkeypatch):
    # 用 fake DockerSandbox(无需真 docker):setup_cmd 返回非零 → RuntimeError
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "m.py").write_text("x = 1\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.t"],
                ["git", "config", "user.name", "t"], ["git", "add", "."],
                ["git", "commit", "-qm", "i"]):
        subprocess.run(cmd, cwd=repo, check=True)

    class FakeBox:
        def __init__(self, workdir, image="python:3.11"):
            self.image = image
            self.name = "fake"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def exec(self, cmd, timeout=120.0):
            if "install -e" in cmd or "setup" in cmd:
                return (1, "could not install deps")
            return (0, "")

    monkeypatch.setattr("anvil_code_agent.eval.runner.has_docker", lambda: True)
    monkeypatch.setattr("anvil_code_agent.eval.runner.DockerSandbox", FakeBox)
    task = Task(id="t", repo=str(repo), prompt="p", verify_cmd="python -m pytest -q")
    with pytest.raises(RuntimeError, match="setup"):
        await solve_task(task, model="deepseek-chat", use_docker=True,
                         setup_cmd="pip install -e .")
