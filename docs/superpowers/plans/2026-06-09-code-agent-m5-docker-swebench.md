# CA-M5 Docker 化 SWE-bench(容器内装依赖跑真实实例)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** 让 SWE-bench 适配器能在 Docker 容器里隔离地装每个真实仓的依赖再跑 agent,把"依赖地狱"关进容器——这是真跑官方 SWE-bench Lite 的前提。复用 M3 DockerSandbox + M4 适配器,只补"容器内装依赖"这一步 + 每实例可选基础镜像。

**Architecture:** `SweInstance` 增 `image`(默认 `python:3.11`,带 gcc/git)+ `install_cmd`(各仓装法,如 `pip install -e .`)。`solve_task` 增 `image`/`setup_cmd` keyword 参:docker 分支起容器(用指定镜像)→ 跑 `setup_cmd` 装仓依赖(失败即 RuntimeError)→ 装 pytest → agent 在容器内跑 → 容器内 verify。swebench CLI 的 `--docker` 把 `inst.image`/`inst.install_cmd` 透传进去。非 docker / 默认行为零变化。

**Tech Stack:** 纯 Python + git/docker CLI;基础镜像 `python:3.11`(buildpack-deps,含编译工具)。

---

## 文件结构

- Modify: `packages/code-agent/src/anvil_code_agent/eval/swebench.py` — SweInstance += image/install_cmd
- Modify: `packages/code-agent/src/anvil_code_agent/eval/runner.py` — solve_task += image/setup_cmd
- Modify: `packages/code-agent/src/anvil_code_agent/cli.py` — swebench --docker 透传
- Modify: `CLAUDE.md` + `examples/06-code-agent/README.md`

---

## Task 1: SweInstance 增 image / install_cmd

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/eval/swebench.py`
- Test: `packages/code-agent/tests/test_swebench_image.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_swebench_image.py
import json

from anvil_code_agent.eval.swebench import SweInstance, load_instances


def test_instance_defaults_image_and_install():
    inst = SweInstance(instance_id="a-1", repo="a/b", base_commit="x",
                       problem_statement="p", test_patch="", fail_to_pass=["t::t"])
    assert inst.image == "python:3.11"
    assert inst.install_cmd == ""


def test_load_reads_image_and_install_cmd(tmp_path):
    f = tmp_path / "inst.jsonl"
    f.write_text(json.dumps({
        "instance_id": "a-1", "repo": "a/b", "base_commit": "x",
        "problem_statement": "p", "test_patch": "P",
        "FAIL_TO_PASS": ["t::t"], "PASS_TO_PASS": [],
        "image": "python:3.9", "install_cmd": "pip install -e .",
    }) + "\n")
    inst = load_instances(str(f))[0]
    assert inst.image == "python:3.9"
    assert inst.install_cmd == "pip install -e ."
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest packages/code-agent/tests/test_swebench_image.py -q`
Expected: FAIL(SweInstance 无 image/install_cmd)

- [ ] **Step 3: 实现**

`swebench.py` 把 `SweInstance` dataclass 增两个带默认值的字段(放在 `pass_to_pass` 之后,保持其在最后或都有默认):

```python
@dataclass
class SweInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str] = field(default_factory=list)
    image: str = "python:3.11"
    install_cmd: str = ""
```

`load_instances` 里构造 `SweInstance(...)` 处追加两行(读 jsonl 的 image/install_cmd,带默认):

```python
                    image=d.get("image", "python:3.11"),
                    install_cmd=d.get("install_cmd", ""),
```

- [ ] **Step 4: 跑测试确认通过 + 回归 swebench 测试**

Run: `uv run pytest packages/code-agent/tests/test_swebench_image.py packages/code-agent/tests/test_swebench_load.py packages/code-agent/tests/test_swebench_adapt.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/swebench.py packages/code-agent/tests/test_swebench_image.py
git commit -m "feat(code-agent): SweInstance gains image + install_cmd (per-instance env)"
```

---

## Task 2: solve_task 增 image / setup_cmd(容器内装依赖)

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/eval/runner.py`
- Test: `packages/code-agent/tests/test_runner_setup.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_runner_setup.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_runner_setup.py -q`
Expected: FAIL(solve_task 无 image/setup_cmd)

- [ ] **Step 3: 实现(改 runner.py 的 solve_task)**

把 `solve_task` 签名与 docker 分支改为(加 image/setup_cmd;docker 分支用 image 起容器、先跑 setup_cmd):

```python
async def solve_task(
    task: Task,
    *,
    model: str,
    max_steps: int = 20,
    use_docker: bool = False,
    image: str | None = None,
    setup_cmd: str | None = None,
) -> RunResult:
    with staged_repo(task.repo) as repo_root:
        with Worktree(repo_root) as wt:
            if use_docker and has_docker():
                with DockerSandbox(wt.path, image=image or "python:3.11") as box:
                    # 1. install the repo's own deps (editable, so agent edits take effect)
                    if setup_cmd:
                        s_rc, s_out = box.exec(setup_cmd, timeout=900.0)
                        if s_rc != 0:
                            raise RuntimeError(
                                f"docker setup failed: setup_cmd rc={s_rc}: {s_out[:300]}"
                            )
                    # 2. ensure pytest is available for the verify step
                    pip_rc, pip_out = box.exec("pip install -q pytest")
                    if pip_rc != 0:
                        raise RuntimeError(
                            f"docker setup failed: pip install pytest rc={pip_rc}: {pip_out[:200]}"
                        )
                    # container runs as root → bind-mounted writes are root-owned (ok for
                    # throwaway worktrees). verify runs in-container; wt.diff() on the host.
                    ctx = ToolContext(workdir=wt.path, executor=box.exec)
                    state = AgentState.new(
                        system=SYSTEM_PROMPT, task=task.prompt, workdir=wt.path, max_steps=max_steps
                    )
                    final = await run(state, model, default_registry(), ctx)
                    rc, out = box.exec(task.verify_cmd)
                    return RunResult(
                        task_id=task.id, passed=rc == 0, steps=final.step, diff=wt.diff()
                    )
            ctx = ToolContext(workdir=wt.path)
            state = AgentState.new(
                system=SYSTEM_PROMPT, task=task.prompt, workdir=wt.path, max_steps=max_steps
            )
            final = await run(state, model, default_registry(), ctx)
            verdict = run_tests({"cmd": task.verify_cmd}, ctx)
            return RunResult(task_id=task.id, passed=verdict.ok, steps=final.step, diff=wt.diff())
```

(注:`DockerSandbox` M3 已支持 `image=` 形参;这里把它显式传入。)

- [ ] **Step 4: 跑测试确认通过 + 回归 docker/runner 测试**

Run: `uv run pytest packages/code-agent/tests/test_runner_setup.py packages/code-agent/tests/test_runner_docker.py packages/code-agent/tests/test_eval_runner.py -q`
Expected: 全 PASS(默认 image/setup_cmd=None → 老 docker 测试行为不变)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/runner.py packages/code-agent/tests/test_runner_setup.py
git commit -m "feat(code-agent): solve_task image + setup_cmd (install repo deps in container)"
```

---

## Task 3: swebench CLI 透传 image / install_cmd

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/cli.py`
- Test: `packages/code-agent/tests/test_cli_swebench_docker.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_cli_swebench_docker.py
from anvil_code_agent.cli import build_parser


def test_swebench_parser_has_docker_flag():
    p = build_parser()
    ns = p.parse_args(["swebench", "--dataset", "i.jsonl", "--docker"])
    assert ns.command == "swebench"
    assert ns.docker is True


def test_swebench_docker_defaults_false():
    p = build_parser()
    ns = p.parse_args(["swebench", "--dataset", "i.jsonl"])
    assert ns.docker is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_cli_swebench_docker.py -q`
Expected: FAIL(swebench 无 --docker)

- [ ] **Step 3: 实现(改 cli.py)**

`build_parser` 的 swebench 子解析器加:

```python
    w.add_argument("--docker", action="store_true", help="run each instance in a Docker sandbox (installs deps in-container)")
```

`_run` 的 swebench 分支里,把 `solve_task(...)` 调用改为透传 docker + 每实例 image/install_cmd:

```python
                res = await solve_task(
                    task,
                    model=ns.model,
                    max_steps=ns.max_steps,
                    use_docker=ns.docker,
                    image=inst.image,
                    setup_cmd=inst.install_cmd or None,
                )
```

- [ ] **Step 4: 跑测试确认通过 + 回归 cli 测试**

Run: `uv run pytest packages/code-agent/tests/test_cli_swebench_docker.py packages/code-agent/tests/test_cli_swebench.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/cli.py packages/code-agent/tests/test_cli_swebench_docker.py
git commit -m "feat(code-agent): swebench --docker passes per-instance image+install_cmd"
```

---

## Task 4: Docker 端到端(合成实例 + 容器内真实装包,skipif 无 docker)

**Files:**
- Test: `packages/code-agent/tests/test_swebench_docker_e2e.py`

- [ ] **Step 1: 写失败/跳过测试**

```python
# packages/code-agent/tests/test_swebench_docker_e2e.py
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
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True).stdout.strip()
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
```

- [ ] **Step 2: 跑测试确认(有 docker 时 PASS,无则 skip)**

Run: `uv run pytest packages/code-agent/tests/test_swebench_docker_e2e.py -q`
Expected: PASS(1 passed)本机 docker 可用;CI skip

- [ ] **Step 3: 实现**

无新代码——本任务证明 Task 1-3 的容器装依赖链路端到端可跑(合成包 editable 安装 → agent 改 → 容器内 verify)。若未过,定位 setup/executor/verify 哪环并修对应实现(不弱化测试)。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_swebench_docker_e2e.py -q`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/tests/test_swebench_docker_e2e.py
git commit -m "test(code-agent): SWE-bench in-Docker e2e (install editable pkg + verify in container)"
```

---

## Task 5: 文档 + 全量回归

**Files:**
- Modify: `anvil/CLAUDE.md`、`examples/06-code-agent/README.md`

- [ ] **Step 1: 全量回归(根目录)**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest -m "not live" -q && uv run ruff check packages/code-agent`
Expected: 全绿(docker 测试本机跑、CI skip),ruff 净

- [ ] **Step 2: 写文档**

`anvil/CLAUDE.md` 的 anvil-code-agent 段补:

```markdown
- `swebench --docker`(M5)— 每实例在 Docker 容器内隔离装依赖再跑:SweInstance 带 `image`(默认 python:3.11,含 gcc/git)+ `install_cmd`(如 `pip install -e .`);solve_task 的 `image`/`setup_cmd` 参在容器起来后先装仓依赖(失败即报错),再 agent 容器内跑 + 容器内 verify。把"依赖地狱"关进容器,是真跑官方 SWE-bench Lite 的前提(不造官方预构建镜像,临场装)
```

`examples/06-code-agent/README.md` 的 CA-M4 SWE-bench 段后补:

```markdown
### CA-M5:Docker 化(容器内装依赖)

真实 SWE-bench 仓各有各的依赖,host 上 ad-hoc 装会互相打架/装不上。CA-M5 把每个实例关进一个容器:

```bash
# 实例 jsonl 每行可带 "image" 和 "install_cmd";--docker 启用容器隔离
uv run anvil-code-agent swebench --dataset swebench_lite.jsonl --limit 5 --docker
```

容器内流程:起 `image` 容器(默认 python:3.11,带编译工具)→ `install_cmd` 装这个仓的依赖(editable,这样 agent 改源即时生效)→ agent 在容器内读写/跑测试 → 容器内跑 FAIL_TO_PASS。装不上的实例如实报 `docker setup failed`,与"代码没修对"区分开。**刻意不造官方每实例预构建镜像**——临场容器装依赖,够拿到一个诚实的真实 pass@1。
```

- [ ] **Step 3: 提交**

```bash
git add CLAUDE.md examples/06-code-agent/README.md
git commit -m "docs(code-agent): CA-M5 Docker-ized SWE-bench (per-instance container deps)"
```

---

## 完成标准(CA-M5 验收)

- `uv run pytest packages/code-agent -m "not live" -q` 全绿(docker e2e 本机跑、CI skip);ruff 净;根目录全量无 collision。
- SweInstance 带 image/install_cmd;solve_task 的 image/setup_cmd 在容器内装依赖(失败 raise),默认 None 时老行为不变。
- Docker 端到端:合成包 editable 安装 → agent 改 → 容器内 verify 转绿。
- swebench --docker 把每实例 image/install_cmd 透传。
- 为真跑官方 SWE-bench Lite 铺好路(编排者随后挑轻量实例实跑出真实 pass@1)。
