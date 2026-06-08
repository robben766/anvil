import shutil
import subprocess
from pathlib import Path

from anvil_code_agent.eval.runner import RunResult, solve_task, staged_repo
from anvil_code_agent.eval.task import Task

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def _init_repo(p):
    subprocess.run(["git", "init", "-q"], cwd=p, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=p, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=p, check=True)
    subprocess.run(["git", "add", "."], cwd=p, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=p, check=True)


async def _fake_fix_agent(state, model, registry, ctx):
    # 直接把 bug 修了,模拟一个成功的 agent;返回的 state 不重要,runner 只看 verify
    calc = Path(ctx.workdir) / "calc.py"
    calc.write_text(calc.read_text().replace("a - b", "a + b"))
    return state.finish("done")


async def _noop_agent(state, model, registry, ctx):
    return state.finish("done")  # 啥也不改 → 测试仍失败


async def test_solve_task_pass_when_agent_fixes(tmp_path, monkeypatch):
    repo = tmp_path / "calc"
    import shutil

    shutil.copytree(GOLDEN / "fixtures" / "calc", repo)
    _init_repo(repo)
    task = Task(id="calc-add", repo=str(repo), prompt="fix", verify_cmd="python -m pytest -q")
    monkeypatch.setattr("anvil_code_agent.eval.runner.run", _fake_fix_agent)
    res = await solve_task(task, model="deepseek-chat")
    assert isinstance(res, RunResult)
    assert res.passed is True


async def test_solve_task_fail_when_agent_noop(tmp_path, monkeypatch):
    repo = tmp_path / "calc"
    import shutil

    shutil.copytree(GOLDEN / "fixtures" / "calc", repo)
    _init_repo(repo)
    task = Task(id="calc-add", repo=str(repo), prompt="fix", verify_cmd="python -m pytest -q")
    monkeypatch.setattr("anvil_code_agent.eval.runner.run", _noop_agent)
    res = await solve_task(task, model="deepseek-chat")
    assert res.passed is False


async def test_pass_rate_aggregates():
    from anvil_code_agent.eval.runner import pass_rate

    results = [
        RunResult(task_id="a", passed=True, steps=3, diff=""),
        RunResult(task_id="b", passed=False, steps=5, diff=""),
    ]
    assert pass_rate(results) == 0.5


def test_staged_repo_non_git_dir_creates_standalone_repo(tmp_path):
    """A non-git dir (like the nested calc fixture) must be staged as a git repo."""
    src = GOLDEN / "fixtures" / "calc"
    dest = tmp_path / "calc_copy"
    shutil.copytree(src, dest)
    # dest is NOT a git repo (no .git), so staged_repo must init one
    with staged_repo(str(dest)) as repo_root:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # The yielded path is its own git root
        import os
        assert os.path.abspath(result.stdout.strip()) == os.path.abspath(repo_root)
        # calc.py is present
        assert (Path(repo_root) / "calc.py").exists()


def test_staged_repo_already_git_repo_yields_unchanged(tmp_path):
    """A directory that is already a standalone git root must be yielded unchanged."""
    repo = tmp_path / "standalone"
    shutil.copytree(GOLDEN / "fixtures" / "calc", repo)
    _init_repo(repo)
    import os
    with staged_repo(str(repo)) as repo_root:
        assert os.path.abspath(repo_root) == os.path.abspath(str(repo))


async def test_solve_task_with_non_git_fixture(tmp_path, monkeypatch):
    """solve_task must succeed even when the fixture dir is NOT a standalone git repo.

    This exercises the staged_repo staging path: calc fixture nested inside the
    anvil repo (not its own git root) gets copied + git-init before Worktree runs.
    """
    # Copy the golden fixture to a plain dir (no .git)
    src = GOLDEN / "fixtures" / "calc"
    dest = tmp_path / "calc_plain"
    shutil.copytree(src, dest)
    task = Task(id="calc-add", repo=str(dest), prompt="fix", verify_cmd="python -m pytest -q")
    monkeypatch.setattr("anvil_code_agent.eval.runner.run", _fake_fix_agent)
    res = await solve_task(task, model="deepseek-chat")
    assert isinstance(res, RunResult)
    assert res.passed is True
