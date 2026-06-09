import subprocess
from pathlib import Path

from anvil_code_agent.eval.runner import solve_task
from anvil_code_agent.eval.swebench import SweInstance, prepare_instance

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
    (p / "m.py").write_text("def f():\n    return 1\n")  # BUG: returns 1, test wants 2
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.t"],
                ["git", "config", "user.name", "t"], ["git", "add", "."],
                ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=p, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True).stdout.strip()
    return p, sha


async def _fake_fix(state, model, registry, ctx, **kw):
    m = Path(ctx.workdir) / "m.py"
    m.write_text(m.read_text().replace("return 1", "return 2"))
    return state.finish("done")


async def test_full_swebench_chain_with_fake_agent(tmp_path, monkeypatch):
    origin, sha = _origin(tmp_path)
    inst = SweInstance(instance_id="m-1", repo=str(origin), base_commit=sha,
                       problem_statement="Make f() return 2 so test_f passes.",
                       test_patch=TEST_PATCH, fail_to_pass=["test_m.py::test_f"])
    task = prepare_instance(inst, str(tmp_path / "work"), repo_url=str(origin))
    monkeypatch.setattr("anvil_code_agent.eval.runner.run", _fake_fix)
    res = await solve_task(task, model="deepseek-chat")
    # 适配器全链:clone→apply test_patch→agent 修→FAIL_TO_PASS 转绿
    assert res.passed is True
    assert res.task_id == "m-1"
