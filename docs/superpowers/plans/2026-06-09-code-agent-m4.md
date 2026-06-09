# CA-M4 SWE-bench 基线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** 把编码 agent 接上 SWE-bench 范式——problem_statement → agent 修 → 跑 FAIL_TO_PASS 测试判定;建实例适配器(确定性合成实例 CI 安全验证)、一组多文件基线 fixture 跑出真实 pass@1、官方 SWE-bench Lite 的 live 手动入口。

**Architecture:** `eval/swebench.py` 把 SWE-bench 实例(instance_id/repo/base_commit/problem_statement/test_patch/FAIL_TO_PASS)适配成现有 `Task`:克隆仓到 base_commit、`git apply` test_patch 并提交(让 Worktree 的 HEAD 含失败测试)、verify_cmd 跑 FAIL_TO_PASS。网络拉取标 live;`apply_test_patch`/`instance_to_task` 纯本地可确定性测。另加两个多文件 bug-fix fixture 进 `baseline.jsonl`,用真模型跑出可复现 pass@1。**不重造官方 Docker harness**(每实例环境构建是 SWE-bench 自己的事)。

**Tech Stack:** 纯 Python + git CLI(clone/checkout/apply via subprocess);复用 M1-M3 runner/工具/沙箱。

---

## 文件结构

- Create: `packages/code-agent/src/anvil_code_agent/eval/swebench.py` — SweInstance + 适配器
- Create: `packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/strops/` — 多文件 fixture①
- Create: `packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/counter/` — 多文件 fixture②
- Create: `packages/code-agent/src/anvil_code_agent/eval/golden/baseline.jsonl` — 基线任务集
- Modify: `packages/code-agent/src/anvil_code_agent/cli.py` — `swebench` 子命令
- Modify: `CLAUDE.md` + `examples/06-code-agent/README.md`

---

## Task 1: SweInstance + load_instances

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/eval/swebench.py`
- Test: `packages/code-agent/tests/test_swebench_load.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_swebench_load.py
import json

from anvil_code_agent.eval.swebench import SweInstance, load_instances


def test_load_handles_list_and_jsonstring_fail_to_pass(tmp_path):
    f = tmp_path / "inst.jsonl"
    rows = [
        {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "abc",
         "problem_statement": "fix it", "test_patch": "PATCH",
         "FAIL_TO_PASS": ["t.py::test_x"], "PASS_TO_PASS": []},
        # 官方 HF 格式:FAIL_TO_PASS 是 JSON 字符串
        {"instance_id": "a__b-2", "repo": "a/b", "base_commit": "def",
         "problem_statement": "fix2", "test_patch": "P2",
         "FAIL_TO_PASS": "[\"t.py::test_y\"]", "PASS_TO_PASS": "[]"},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    insts = load_instances(str(f))
    assert len(insts) == 2
    assert isinstance(insts[0], SweInstance)
    assert insts[0].fail_to_pass == ["t.py::test_x"]
    assert insts[1].fail_to_pass == ["t.py::test_y"]  # 字符串也解析成 list
    assert insts[0].repo == "a/b"
    assert insts[0].base_commit == "abc"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest packages/code-agent/tests/test_swebench_load.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/eval/swebench.py
"""Adapt a SWE-bench(-Lite) instance to anvil's bug-fix Task: clone the repo at
base_commit, git-apply the test_patch (which adds the failing tests) and commit it so a
worktree's HEAD carries it, then verify the FAIL_TO_PASS tests. We do NOT reproduce the
official per-instance Docker harness — environment build is the benchmark's own concern."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

from anvil_code_agent.eval.task import Task


def _as_list(v: object) -> list[str]:
    if isinstance(v, str):
        return list(json.loads(v))
    return list(v) if v else []


@dataclass
class SweInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str] = field(default_factory=list)


def load_instances(path: str) -> list[SweInstance]:
    out: list[SweInstance] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                SweInstance(
                    instance_id=d["instance_id"],
                    repo=d["repo"],
                    base_commit=d["base_commit"],
                    problem_statement=d["problem_statement"],
                    test_patch=d.get("test_patch", ""),
                    fail_to_pass=_as_list(d.get("FAIL_TO_PASS", [])),
                    pass_to_pass=_as_list(d.get("PASS_TO_PASS", [])),
                )
            )
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_swebench_load.py -q`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/swebench.py packages/code-agent/tests/test_swebench_load.py
git commit -m "feat(code-agent): SweInstance + load_instances (handles HF json-string fields)"
```

---

## Task 2: apply_test_patch + instance_to_task

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/eval/swebench.py`(追加)
- Test: `packages/code-agent/tests/test_swebench_adapt.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_swebench_adapt.py
import subprocess

from anvil_code_agent.eval.swebench import SweInstance, apply_test_patch, instance_to_task
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_swebench_adapt.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现(追加到 swebench.py)**

```python
# 追加到 packages/code-agent/src/anvil_code_agent/eval/swebench.py 末尾


def apply_test_patch(repo_root: str, instance: SweInstance) -> None:
    """git-apply the instance's test_patch and commit it, so a worktree checked out at
    HEAD carries the (currently failing) tests. Raises RuntimeError if the patch fails."""
    if not instance.test_patch.strip():
        return
    r = subprocess.run(
        ["git", "apply", "-"],
        cwd=repo_root,
        input=instance.test_patch,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git apply test_patch failed: {r.stderr.strip()}")
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", f"apply test_patch for {instance.instance_id}"],
        cwd=repo_root,
        check=True,
    )


def instance_to_task(instance: SweInstance, repo_root: str) -> Task:
    """Build a bug-fix Task: agent gets the problem statement; success = FAIL_TO_PASS pass."""
    targets = " ".join(instance.fail_to_pass)
    return Task(
        id=instance.instance_id,
        repo=repo_root,
        prompt=instance.problem_statement,
        verify_cmd=f"python -m pytest {targets} -q",
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_swebench_adapt.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/swebench.py packages/code-agent/tests/test_swebench_adapt.py
git commit -m "feat(code-agent): apply_test_patch (commit failing tests) + instance_to_task"
```

---

## Task 3: fetch_repo + prepare_instance(network,live)

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/eval/swebench.py`(追加)
- Test: `packages/code-agent/tests/test_swebench_prepare.py`

- [ ] **Step 1: 写失败测试(本地仓模拟 clone 源,不碰真 GitHub)**

```python
# packages/code-agent/tests/test_swebench_prepare.py
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
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=p, capture_output=True, text=True).stdout.strip()
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_swebench_prepare.py -q`
Expected: FAIL(ImportError: prepare_instance)

- [ ] **Step 3: 实现(追加到 swebench.py)**

```python
# 追加到 packages/code-agent/src/anvil_code_agent/eval/swebench.py 末尾


def fetch_repo(instance: SweInstance, dest: str, *, repo_url: str | None = None) -> None:
    """Clone the instance repo to dest and check out base_commit. repo_url defaults to
    GitHub (https://github.com/{repo}.git); pass a local path/URL to avoid the network."""
    url = repo_url or f"https://github.com/{instance.repo}.git"
    subprocess.run(["git", "clone", "--quiet", url, dest], check=True)
    subprocess.run(["git", "checkout", "-q", instance.base_commit], cwd=dest, check=True)


def prepare_instance(
    instance: SweInstance, dest: str, *, repo_url: str | None = None
) -> Task:
    """Full setup: fetch repo @ base_commit → apply+commit test_patch → build Task."""
    fetch_repo(instance, dest, repo_url=repo_url)
    apply_test_patch(dest, instance)
    return instance_to_task(instance, dest)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_swebench_prepare.py -q`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/swebench.py packages/code-agent/tests/test_swebench_prepare.py
git commit -m "feat(code-agent): fetch_repo + prepare_instance (repo_url injectable for tests)"
```

---

## Task 4: 合成实例端到端(适配器全链,确定性,无网络)

**Files:**
- Test: `packages/code-agent/tests/test_swebench_e2e.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_swebench_e2e.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_swebench_e2e.py -q`
Expected: 初次应 FAIL(若 solve_task 对 prepared repo 工作正常则直接 PASS——确认是真跑通而非假绿:先临时把 `_fake_fix` 改成不改文件,应得 res.passed=False,再改回)

- [ ] **Step 3: 实现**

无新代码——本任务是适配器全链的集成测试,验证 Task 1-3 的部件串起来能跑通。若测试未过,定位是 apply/Worktree/verify 哪一环并修对应实现(不弱化测试)。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_swebench_e2e.py -q`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/tests/test_swebench_e2e.py
git commit -m "test(code-agent): synthetic SWE-bench instance end-to-end (no network)"
```

---

## Task 5: 多文件基线 fixture + baseline.jsonl

**Files:**
- Create: `.../eval/golden/fixtures/strops/strops/__init__.py`、`strops/core.py`、`test_strops.py`
- Create: `.../eval/golden/fixtures/counter/counter/__init__.py`、`counter/logic.py`、`test_counter.py`
- Create: `.../eval/golden/baseline.jsonl`
- Test: `packages/code-agent/tests/test_baseline_fixtures.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_baseline_fixtures.py
import shutil
import subprocess
from pathlib import Path

from anvil_code_agent.eval.task import load_tasks

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def test_baseline_jsonl_loads_multifile_tasks():
    tasks = load_tasks(str(GOLDEN / "baseline.jsonl"))
    ids = {t.id for t in tasks}
    assert {"strops-upper", "counter-evens"} <= ids


def _fails_before_passes_after(tmp_path, fixture, bug, fix):
    dst = tmp_path / fixture
    shutil.copytree(GOLDEN / "fixtures" / fixture, dst)
    before = subprocess.run(["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True)
    assert before.returncode != 0  # 带 bug → 失败
    # 在所有 .py 里把 bug 改成 fix
    for py in dst.rglob("*.py"):
        txt = py.read_text()
        if bug in txt:
            py.write_text(txt.replace(bug, fix))
    after = subprocess.run(["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True)
    assert after.returncode == 0  # 修后 → 通过


def test_strops_fixture_wellformed(tmp_path):
    _fails_before_passes_after(tmp_path, "strops", "s.lower()", "s.upper()")


def test_counter_fixture_wellformed(tmp_path):
    _fails_before_passes_after(tmp_path, "counter", "n % 2 == 1", "n % 2 == 0")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_baseline_fixtures.py -q`
Expected: FAIL(文件不存在)

- [ ] **Step 3: 创建 fixtures + baseline.jsonl**

`strops/strops/__init__.py`:
```python
from strops.core import shout

__all__ = ["shout"]
```
`strops/strops/core.py`:
```python
def shout(s):
    return s.lower() + "!"  # BUG: should upper-case
```
`strops/test_strops.py`:
```python
from strops import shout


def test_shout():
    assert shout("hi") == "HI!"
```

`counter/counter/__init__.py`:
```python
from counter.logic import count_evens

__all__ = ["count_evens"]
```
`counter/counter/logic.py`:
```python
def count_evens(nums):
    return sum(1 for n in nums if n % 2 == 1)  # BUG: counts odds
```
`counter/test_counter.py`:
```python
from counter import count_evens


def test_count_evens():
    assert count_evens([1, 2, 3, 4]) == 2
```

`baseline.jsonl`(每行一个 Task;repo 相对 dataset 目录解析):
```jsonl
{"id": "calc-add", "repo": "fixtures/calc", "prompt": "The test_add test is failing. Find and fix the bug in calc.py, then run the tests until they pass.", "verify_cmd": "python -m pytest -q"}
{"id": "strops-upper", "repo": "fixtures/strops", "prompt": "test_shout is failing. The shout() function should upper-case its input and append '!'. Locate the function across the package, fix it, and make the tests pass.", "verify_cmd": "python -m pytest -q"}
{"id": "counter-evens", "repo": "fixtures/counter", "prompt": "test_count_evens is failing. count_evens should count EVEN numbers. Find the bug in the package and fix it until tests pass.", "verify_cmd": "python -m pytest -q"}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_baseline_fixtures.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/strops packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/counter packages/code-agent/src/anvil_code_agent/eval/golden/baseline.jsonl packages/code-agent/tests/test_baseline_fixtures.py
git commit -m "feat(code-agent): multi-file baseline fixtures (strops/counter) + baseline.jsonl"
```

---

## Task 6: CLI swebench 子命令

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/cli.py`
- Test: `packages/code-agent/tests/test_cli_swebench.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_cli_swebench.py
from anvil_code_agent.cli import build_parser


def test_parser_has_swebench():
    p = build_parser()
    ns = p.parse_args(["swebench", "--dataset", "inst.jsonl", "--limit", "3", "--workdir", "/tmp/wb"])
    assert ns.command == "swebench"
    assert ns.dataset == "inst.jsonl"
    assert ns.limit == 3
    assert ns.workdir == "/tmp/wb"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_cli_swebench.py -q`
Expected: FAIL(swebench 子命令不存在)

- [ ] **Step 3: 实现(改 cli.py)**

`cli.py` 顶部 import 区追加:

```python
from anvil_code_agent.eval.swebench import load_instances, prepare_instance
```

`build_parser` 里追加 swebench 子解析器(放在 eval 之后):

```python
    w = sub.add_parser("swebench", help="run SWE-bench-format instances (clones repos; live)")
    w.add_argument("--dataset", required=True, help="jsonl of SWE-bench instances")
    w.add_argument("--limit", type=int, default=0, help="max instances (0 = all)")
    w.add_argument("--workdir", default="/tmp/anvil-swebench", help="dir to clone repos into")
    w.add_argument("--model", default="deepseek-chat")
    w.add_argument("--max-steps", type=int, default=40)
```

`_run` 里在 `eval` 分支之后、`return` 之前追加 swebench 分支:

```python
    if ns.command == "swebench":
        import os

        instances = load_instances(ns.dataset)
        if ns.limit:
            instances = instances[: ns.limit]
        results = []
        for inst in instances:
            dest = os.path.join(ns.workdir, inst.instance_id)
            try:
                task = prepare_instance(inst, dest)
                res = await solve_task(task, model=ns.model, max_steps=ns.max_steps)
                print(f"  {res.task_id}: {'PASS' if res.passed else 'FAIL'} ({res.steps} steps)")
                results.append(res)
            except Exception as e:  # noqa: BLE001 — one broken instance must not kill the run
                print(f"  {inst.instance_id}: ERROR ({e})")
                results.append(RunResult(task_id=inst.instance_id, passed=False, steps=0, diff=""))
        rate = pass_rate(results)
        print(f"SWE-bench pass@1: {rate:.2%} ({sum(r.passed for r in results)}/{len(results)})")
        return 0
```

(确保 `cli.py` 已 import `RunResult`——若没有,在 `from anvil_code_agent.eval.runner import ...` 行补上 `RunResult`。)

- [ ] **Step 4: 跑测试确认通过 + 回归 cli 测试**

Run: `uv run pytest packages/code-agent/tests/test_cli_swebench.py packages/code-agent/tests/test_cli.py packages/code-agent/tests/test_cli_eval.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/cli.py packages/code-agent/tests/test_cli_swebench.py
git commit -m "feat(code-agent): CLI swebench subcommand (prepare+solve+pass@1)"
```

---

## Task 7: 文档 + 全量回归(真实 pass@1 数字由编排者补)

**Files:**
- Modify: `anvil/CLAUDE.md`、`examples/06-code-agent/README.md`

- [ ] **Step 1: 全量回归(根目录,查 collision)**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest -m "not live" -q && uv run ruff check packages/code-agent`
Expected: 全绿,ruff 净

- [ ] **Step 2: 写文档**

`anvil/CLAUDE.md` 的 anvil-code-agent 段补:

```markdown
- `eval/swebench.py`(M4)— SWE-bench(-Lite)实例适配器:SweInstance/load_instances(兼容 HF json-string 字段)/fetch_repo(clone@base_commit)/apply_test_patch(git apply 并提交,让 worktree HEAD 含失败测试)/instance_to_task(verify=跑 FAIL_TO_PASS)/prepare_instance(全链);**不重造官方 Docker harness**
- `eval/golden/baseline.jsonl`(M4)— calc/strops/counter 三个 bug-fix 任务(后两个多文件,逼 agent 用 repo_map/grep 定位)
- CLI: `anvil-code-agent eval --dataset .../baseline.jsonl`(本地基线 pass@1)/ `anvil-code-agent swebench --dataset <instances.jsonl> [--limit N]`(官方实例,clone 真实仓,live)
```

`examples/06-code-agent/README.md` 末尾追加(`<RATE>`/`<N>` 由编排者跑真实基线后回填):

```markdown
## CA-M4:SWE-bench 基线

**本地可复现基线**(三个 bug-fix 任务,后两个多文件):
```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
uv run anvil-code-agent eval --dataset packages/code-agent/src/anvil_code_agent/eval/golden/baseline.jsonl
```
真实跑分:**pass@1 = <RATE>(<N>/3)**(deepseek-chat 驱动)。

**接官方 SWE-bench Lite**(live,拉真实仓):
```bash
# 取官方实例 jsonl(princeton-nlp/SWE-bench_Lite),然后:
uv run anvil-code-agent swebench --dataset swebench_lite.jsonl --limit 5
```
适配器做的事:clone 仓到 base_commit → `git apply` test_patch 并提交(失败测试进 HEAD)→ agent 修 → 跑 FAIL_TO_PASS 判定 pass@1。**刻意不重造官方每实例 Docker 环境构建**——那是 SWE-bench 自己的 harness 范畴;本里程碑触底的是"problem statement → agent → FAIL_TO_PASS 判定"这条评测范式。
```

- [ ] **Step 3: 提交**

```bash
git add CLAUDE.md examples/06-code-agent/README.md
git commit -m "docs(code-agent): CA-M4 SWE-bench adapter + baseline (milestone complete)"
```

---

## 完成标准(CA-M4 验收)

- `uv run pytest packages/code-agent -m "not live" -q` 全绿;ruff 净;根目录全量无 collision。
- 适配器:load_instances 兼容 HF json-string;apply_test_patch 提交失败测试;合成实例端到端(无网络)跑通 clone→apply→修→FAIL_TO_PASS 转绿。
- baseline.jsonl 三任务(多文件 fixture 设计正确:带 bug 失败、修后通过);CLI `swebench` 子命令解析 + 全链可调。
- 编排者用真模型跑 baseline 得到真实 pass@1 并回填 README。
- 触底:SWE-bench 评测范式(problem statement → agent → FAIL_TO_PASS 判定),不重造官方 Docker harness。
