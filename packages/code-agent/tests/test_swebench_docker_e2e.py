import subprocess
from pathlib import Path

import pytest
from anvil_code_agent.eval.runner import solve_task
from anvil_code_agent.eval.swebench import SweInstance, prepare_instance
from anvil_code_agent.sandbox import has_docker

pytestmark = pytest.mark.skipif(not has_docker(), reason="docker daemon unavailable")

# test_patch 新增一个导入已安装包的失败测试
TEST_PATCH = (
    "diff --git a/test_pkg.py b/test_pkg.py\n"
    "new file mode 100644\n"
    "index 0000000..1111111\n"
    "--- /dev/null\n"
    "+++ b/test_pkg.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+from synthpkg import f\n"
    "+def test_f(): assert f() == 2\n"
)


def _origin(tmp_path):
    p = tmp_path / "origin"
    p.mkdir()
    (p / "synthpkg.py").write_text("def f():\n    return 1\n")  # BUG: returns 1
    (p / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='synthpkg', version='0.1', py_modules=['synthpkg'])\n"
    )
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.t"],
                ["git", "config", "user.name", "t"], ["git", "add", "."],
                ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=p, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True
    ).stdout.strip()
    return p, sha


async def _fake_fix(state, model, registry, ctx, **kw):
    pkg = Path(ctx.workdir) / "synthpkg.py"
    pkg.write_text(pkg.read_text().replace("return 1", "return 2"))
    return state.finish("done")


async def test_swebench_in_docker_installs_and_verifies(tmp_path, monkeypatch):
    origin, sha = _origin(tmp_path)
    inst = SweInstance(
        instance_id="synth-1", repo=str(origin), base_commit=sha,
        problem_statement="Make synthpkg.f() return 2 so test_f passes.",
        test_patch=TEST_PATCH, fail_to_pass=["test_pkg.py::test_f"],
        image="python:3.11", install_cmd="pip install -e . --no-build-isolation -q",
    )
    task = prepare_instance(inst, str(tmp_path / "work"), repo_url=str(origin))
    monkeypatch.setattr("anvil_code_agent.eval.runner.run", _fake_fix)
    # 容器内:装 synthpkg(editable)→ agent 改源 → 容器内 pytest 跑 FAIL_TO_PASS
    res = await solve_task(
        task, model="deepseek-chat", use_docker=True,
        image=inst.image, setup_cmd=inst.install_cmd,
    )
    assert res.passed is True
