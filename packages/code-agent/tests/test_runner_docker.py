# packages/code-agent/tests/test_runner_docker.py
import inspect
import shutil
import subprocess
from pathlib import Path

import pytest
from anvil_code_agent.eval.runner import solve_task
from anvil_code_agent.eval.task import Task
from anvil_code_agent.sandbox import has_docker

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def _init_repo(p: Path) -> None:
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(cmd, cwd=p, check=True)


def test_solve_task_accepts_use_docker_param():
    # 签名暴露 use_docker(默认 False,保持 host 路径)
    sig = inspect.signature(solve_task)
    assert "use_docker" in sig.parameters
    assert sig.parameters["use_docker"].default is False


async def test_docker_pip_install_failure_raises_runtime_error(tmp_path, monkeypatch):
    """If pip install pytest fails inside the container, solve_task must raise
    RuntimeError mentioning 'docker setup failed', not silently return passed=False."""

    # Build a minimal git repo so Worktree context works without real docker.
    repo = tmp_path / "calc"
    shutil.copytree(GOLDEN / "fixtures" / "calc", repo)
    _init_repo(repo)

    # Fake sandbox: pip install returns rc=1 (network unreachable).
    class FakeDockerSandbox:
        def __init__(self, *args, **kwargs):
            self.name = "fake-box"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def exec(self, cmd: str, **kwargs):  # noqa: A003
            if "pip install" in cmd:
                return (1, "network unreachable")
            return (0, "")

    # Patch DockerSandbox and has_docker in the runner module.
    monkeypatch.setattr("anvil_code_agent.eval.runner.DockerSandbox", FakeDockerSandbox)
    monkeypatch.setattr("anvil_code_agent.eval.runner.has_docker", lambda: True)

    # Fake agent that does nothing (should never reach verify anyway).
    async def _noop_agent(state, model, registry, ctx, **kw):
        return state.finish("done")

    monkeypatch.setattr("anvil_code_agent.eval.runner.run", _noop_agent)

    task = Task(id="calc-add", repo=str(repo), prompt="fix", verify_cmd="python -m pytest -q")

    with pytest.raises(RuntimeError, match="docker setup failed"):
        await solve_task(task, model="deepseek-chat", use_docker=True)


@pytest.mark.skipif(not has_docker(), reason="docker daemon unavailable")
async def test_solve_task_with_docker_executes_in_container(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"
    repo = tmp_path / "calc"
    shutil.copytree(GOLDEN / "fixtures" / "calc", repo)
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.t"],
                ["git", "config", "user.name", "t"], ["git", "add", "."],
                ["git", "commit", "-qm", "i"]):
        subprocess.run(cmd, cwd=repo, check=True)

    async def _fake_fix(state, model, registry, ctx, **kw):
        # 断言 ctx.executor 已接上(docker 模式)
        assert ctx.executor is not None
        calc = Path(ctx.workdir) / "calc.py"
        calc.write_text(calc.read_text().replace("a - b", "a + b"))
        return state.finish("done")

    monkeypatch.setattr("anvil_code_agent.eval.runner.run", _fake_fix)
    task = Task(id="calc-add", repo=str(repo), prompt="fix", verify_cmd="python -m pytest -q")
    res = await solve_task(task, model="deepseek-chat", use_docker=True)
    assert res.passed is True
