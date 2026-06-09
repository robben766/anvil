# packages/code-agent/tests/test_runner_docker.py
import inspect
import subprocess

import pytest
from anvil_code_agent.eval.runner import solve_task
from anvil_code_agent.sandbox import has_docker
from anvil_code_agent.eval.task import Task


def test_solve_task_accepts_use_docker_param():
    # 签名暴露 use_docker(默认 False,保持 host 路径)
    sig = inspect.signature(solve_task)
    assert "use_docker" in sig.parameters
    assert sig.parameters["use_docker"].default is False


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
