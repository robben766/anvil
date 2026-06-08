# CA-M1 最小可用循环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自研编码 agent 的最小可用循环——一个 reducer-style while 循环驱动 LLM,用 read/edit/bash/run_tests 四工具在 git worktree 隔离内端到端修复真 bug,并由 eval runner 给出 pass 率。

**Architecture:** agent 即无状态 reducer:`step(state, registry) -> state'` 做一次 gateway tool_use 往返并执行返回的工具调用;`run()` 循环 step 直到模型声明完成或守护触发。工具实现 ACI 理念,失败也返回可读结果喂回模型。复用 anvil-gateway(已验证 tool_use 往返)/ anvil-obs(span 追踪)/ anvil-eval(基建)。

**Tech Stack:** Python 3.12,uv workspace,hatchling;httpx/respx(测试 mock gateway);pytest + pytest-asyncio;git worktree;ruff。

---

## 文件结构

- Create: `packages/code-agent/pyproject.toml` — 包定义(name=anvil-code-agent)
- Create: `packages/code-agent/src/anvil_code_agent/__init__.py` — 导出
- Create: `packages/code-agent/src/anvil_code_agent/state.py` — `AgentState`
- Create: `packages/code-agent/src/anvil_code_agent/tools/base.py` — `Tool`/`ToolResult`/`ToolContext`/`ToolRegistry`
- Create: `packages/code-agent/src/anvil_code_agent/tools/fs.py` — `read_file`/`edit_file`
- Create: `packages/code-agent/src/anvil_code_agent/tools/shell.py` — `bash`
- Create: `packages/code-agent/src/anvil_code_agent/tools/verify.py` — `run_tests`
- Create: `packages/code-agent/src/anvil_code_agent/sandbox.py` — worktree
- Create: `packages/code-agent/src/anvil_code_agent/harness/loop.py` — reducer 循环
- Create: `packages/code-agent/src/anvil_code_agent/eval/task.py` — `Task`
- Create: `packages/code-agent/src/anvil_code_agent/eval/runner.py` — runner + pass 率
- Create: `packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/` — bug-fix 任务集
- Create: `packages/code-agent/src/anvil_code_agent/cli.py` — CLI
- Modify: `pyproject.toml`(根)— workspace members 加 `packages/code-agent`

---

## Task 1: 包骨架 + workspace 注册

**Files:**
- Create: `packages/code-agent/pyproject.toml`
- Create: `packages/code-agent/src/anvil_code_agent/__init__.py`
- Modify: `pyproject.toml`(根,`members` 行)
- Test: `packages/code-agent/tests/test_code_agent_sanity.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_code_agent_sanity.py
def test_package_imports():
    import anvil_code_agent

    assert anvil_code_agent.__version__ == "0.1.0"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest packages/code-agent -q`
Expected: FAIL(模块不存在 / 包未安装)

- [ ] **Step 3: 写 pyproject + __init__ + 注册 workspace**

```toml
# packages/code-agent/pyproject.toml
[project]
name = "anvil-code-agent"
version = "0.1.0"
description = "anvil: self-built coding agent harness — reducer loop + ACI tools + closed-loop verify"
requires-python = ">=3.12"
dependencies = [
    "anvil-gateway",
    "anvil-obs",
    "anvil-eval",
]

[project.scripts]
anvil-code-agent = "anvil_code_agent.cli:main"

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "respx>=0.21",
    "httpx>=0.27",
    "ruff>=0.6",
]

[tool.uv.sources]
anvil-gateway = { workspace = true }
anvil-obs = { workspace = true }
anvil-eval = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/anvil_code_agent"]
```

```python
# packages/code-agent/src/anvil_code_agent/__init__.py
"""anvil-code-agent: self-built coding agent harness (圈3 agent)."""

__version__ = "0.1.0"
```

根 `pyproject.toml` 把 `members = ["packages/core/*", "packages/kb", "apps/kb-api"]`
改为 `members = ["packages/core/*", "packages/kb", "packages/code-agent", "apps/kb-api"]`。

同时创建 `packages/code-agent/pytest.ini`(贴合现有 async 测试):

```ini
# packages/code-agent/pytest.ini
[pytest]
asyncio_mode = auto
markers =
    live: 需要真实 API key,手动运行
    slow: 慢任务(沙箱/SWE-bench)
```

- [ ] **Step 4: 同步并跑测试确认通过**

Run: `cd /home/itachi/workspace/ai/anvil && uv sync --all-packages && uv run pytest packages/code-agent -q`
Expected: PASS(1 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent pyproject.toml
git commit -m "feat(code-agent): package scaffold + workspace registration"
```

---

## Task 2: AgentState 不可变快照

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/state.py`
- Test: `packages/code-agent/tests/test_state.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_state.py
from anvil_code_agent.state import AgentState


def test_initial_state():
    s = AgentState.new(system="be a coder", task="fix bug", workdir="/tmp/wt", max_steps=10)
    assert s.step == 0
    assert s.status == "running"
    assert s.messages[0] == {"role": "system", "content": "be a coder"}
    assert s.messages[1] == {"role": "user", "content": "fix bug"}


def test_with_appended_is_immutable():
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=5)
    s2 = s.append({"role": "assistant", "content": "hi"}).advance()
    assert s.step == 0 and len(s.messages) == 2  # 原对象不变
    assert s2.step == 1 and len(s2.messages) == 3


def test_finish_sets_status():
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=5)
    assert s.finish("done").status == "done"
    assert s.finish("exhausted").status == "exhausted"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_state.py -q`
Expected: FAIL(ImportError: AgentState)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/state.py
"""AgentState: immutable snapshot of one agent run. Reducer steps return new states."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

Message = dict[str, Any]
Status = str  # "running" | "done" | "exhausted" | "error"


@dataclass(frozen=True)
class AgentState:
    messages: tuple[Message, ...]
    step: int
    max_steps: int
    workdir: str
    status: Status = "running"

    @classmethod
    def new(cls, *, system: str, task: str, workdir: str, max_steps: int) -> AgentState:
        return cls(
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ),
            step=0,
            max_steps=max_steps,
            workdir=workdir,
            status="running",
        )

    def append(self, *msgs: Message) -> AgentState:
        return replace(self, messages=self.messages + tuple(msgs))

    def advance(self) -> AgentState:
        return replace(self, step=self.step + 1)

    def finish(self, status: Status) -> AgentState:
        return replace(self, status=status)
```

(注:`field` 导入保留备用即可;若 ruff 报未使用则删除 `field` 导入。)

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_state.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/state.py packages/code-agent/tests/test_state.py
git commit -m "feat(code-agent): immutable AgentState reducer snapshot"
```

---

## Task 3: 工具协议 + 注册表

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/tools/__init__.py`(空)
- Create: `packages/code-agent/src/anvil_code_agent/tools/base.py`
- Test: `packages/code-agent/tests/test_tools_base.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_tools_base.py
import pytest
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool


def test_tool_decorator_builds_schema():
    @tool(name="echo", description="echo text", params={"text": {"type": "string"}}, required=["text"])
    def echo(args, ctx):
        return ToolResult(content=args["text"], ok=True)

    assert echo.name == "echo"
    sch = echo.schema
    assert sch["type"] == "function"
    assert sch["function"]["name"] == "echo"
    assert sch["function"]["parameters"]["required"] == ["text"]


def test_registry_collects_schemas_and_dispatches():
    @tool(name="echo", description="d", params={"text": {"type": "string"}}, required=["text"])
    def echo(args, ctx):
        return ToolResult(content=args["text"].upper(), ok=True)

    reg = ToolRegistry([echo])
    assert [s["function"]["name"] for s in reg.schemas()] == ["echo"]
    res = reg.dispatch("echo", {"text": "hi"}, ToolContext(workdir="/tmp"))
    assert res.content == "HI" and res.ok


def test_registry_unknown_tool_returns_error_result():
    reg = ToolRegistry([])
    res = reg.dispatch("nope", {}, ToolContext(workdir="/tmp"))
    assert res.ok is False
    assert "unknown tool" in res.content.lower()


def test_dispatch_catches_tool_exception_as_error_result():
    @tool(name="boom", description="d", params={}, required=[])
    def boom(args, ctx):
        raise RuntimeError("kaboom")

    reg = ToolRegistry([boom])
    res = reg.dispatch("boom", {}, ToolContext(workdir="/tmp"))
    assert res.ok is False
    assert "kaboom" in res.content
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_tools_base.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/tools/__init__.py
```

```python
# packages/code-agent/src/anvil_code_agent/tools/base.py
"""Tool protocol + registry. ACI principle: tools ALWAYS return a readable result,
even on failure — the failure text is feedback the model uses to retry (12-Factor #9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    content: str
    ok: bool
    truncated: bool = False


@dataclass
class ToolContext:
    workdir: str
    timeout: float = 120.0
    max_output: int = 4096


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        params: dict[str, Any],
        required: list[str],
        fn: Callable[[dict[str, Any], ToolContext], ToolResult],
    ):
        self.name = name
        self.description = description
        self._fn = fn
        self.schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        }

    def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._fn(args, ctx)


def tool(*, name: str, description: str, params: dict[str, Any], required: list[str]):
    def deco(fn: Callable[[dict[str, Any], ToolContext], ToolResult]) -> Tool:
        return Tool(name, description, params, required, fn)

    return deco


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def dispatch(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t = self._tools.get(name)
        if t is None:
            return ToolResult(content=f"unknown tool: {name}", ok=False)
        try:
            return t(args, ctx)
        except Exception as e:  # noqa: BLE001 — failures are feedback, never crash the loop
            return ToolResult(content=f"tool '{name}' raised {type(e).__name__}: {e}", ok=False)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_tools_base.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools packages/code-agent/tests/test_tools_base.py
git commit -m "feat(code-agent): tool protocol + registry (failures as feedback)"
```

---

## Task 4: fs.read_file 工具

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/tools/fs.py`
- Test: `packages/code-agent/tests/test_tools_fs_read.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_tools_fs_read.py
from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.fs import read_file


def test_read_returns_numbered_lines(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    res = read_file({"path": "a.py"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "1\tx = 1" in res.content
    assert "2\ty = 2" in res.content


def test_read_missing_file_is_error(tmp_path):
    res = read_file({"path": "nope.py"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok is False
    assert "not found" in res.content.lower()


def test_read_truncates_long_file(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line{i}" for i in range(10000)))
    res = read_file({"path": "big.txt"}, ToolContext(workdir=str(tmp_path), max_output=200))
    assert res.truncated
    assert len(res.content) <= 400  # 截断 + 提示行的宽松上界
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_tools_fs_read.py -q`
Expected: FAIL(ImportError: read_file)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/tools/fs.py
"""Filesystem tools: read_file (numbered lines + truncation) and edit_file
(SEARCH-REPLACE with strict guardrails — editing reliability决定可用性)."""

from __future__ import annotations

import os

from anvil_code_agent.tools.base import ToolContext, ToolResult, tool


def _resolve(workdir: str, path: str) -> str:
    return os.path.join(workdir, path)


@tool(
    name="read_file",
    description="Read a file relative to the working dir. Returns lines prefixed with line numbers.",
    params={
        "path": {"type": "string", "description": "file path relative to working dir"},
    },
    required=["path"],
)
def read_file(args: dict, ctx: ToolContext) -> ToolResult:
    full = _resolve(ctx.workdir, args["path"])
    if not os.path.isfile(full):
        return ToolResult(content=f"file not found: {args['path']}", ok=False)
    with open(full, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    numbered = "\n".join(f"{i + 1}\t{ln}" for i, ln in enumerate(text.splitlines()))
    if len(numbered) > ctx.max_output:
        head = numbered[: ctx.max_output]
        return ToolResult(
            content=head + f"\n... [truncated, {len(numbered)} chars total]",
            ok=True,
            truncated=True,
        )
    return ToolResult(content=numbered, ok=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_tools_fs_read.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools/fs.py packages/code-agent/tests/test_tools_fs_read.py
git commit -m "feat(code-agent): read_file tool (numbered lines + truncation)"
```

---

## Task 5: fs.edit_file 工具(SEARCH-REPLACE + 护栏)— 可用性关键

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/tools/fs.py`(追加 `edit_file`)
- Test: `packages/code-agent/tests/test_tools_fs_edit.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_tools_fs_edit.py
from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.fs import edit_file


def _ctx(p):
    return ToolContext(workdir=str(p))


def test_edit_applies_unique_match(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def add(a, b):\n    return a - b\n")
    res = edit_file(
        {"path": "a.py", "search": "    return a - b", "replace": "    return a + b"},
        _ctx(tmp_path),
    )
    assert res.ok
    assert f.read_text() == "def add(a, b):\n    return a + b\n"


def test_edit_not_found_is_error_and_no_write(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    res = edit_file({"path": "a.py", "search": "y = 2", "replace": "y = 3"}, _ctx(tmp_path))
    assert res.ok is False
    assert "not found" in res.content.lower()
    assert f.read_text() == "x = 1\n"  # 未改


def test_edit_multiple_matches_is_error(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v = 1\nv = 1\n")
    res = edit_file({"path": "a.py", "search": "v = 1", "replace": "v = 2"}, _ctx(tmp_path))
    assert res.ok is False
    assert "multiple" in res.content.lower() or "2 matches" in res.content.lower()
    assert f.read_text() == "v = 1\nv = 1\n"


def test_edit_empty_search_is_error(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    res = edit_file({"path": "a.py", "search": "", "replace": "y = 2"}, _ctx(tmp_path))
    assert res.ok is False
    assert "empty" in res.content.lower()


def test_edit_missing_file_is_error(tmp_path):
    res = edit_file({"path": "no.py", "search": "a", "replace": "b"}, _ctx(tmp_path))
    assert res.ok is False
    assert "not found" in res.content.lower()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_tools_fs_edit.py -q`
Expected: FAIL(ImportError: edit_file)

- [ ] **Step 3: 实现(追加到 fs.py)**

```python
# 追加到 packages/code-agent/src/anvil_code_agent/tools/fs.py 末尾


@tool(
    name="edit_file",
    description=(
        "Replace an EXACT unique snippet in a file. 'search' must match exactly once "
        "(whitespace-sensitive). If it matches zero or multiple times, the edit is rejected "
        "and you must read the file and provide a more specific search block."
    ),
    params={
        "path": {"type": "string", "description": "file path relative to working dir"},
        "search": {"type": "string", "description": "exact snippet to find (must be unique)"},
        "replace": {"type": "string", "description": "replacement snippet"},
    },
    required=["path", "search", "replace"],
)
def edit_file(args: dict, ctx: ToolContext) -> ToolResult:
    path, search, replace = args["path"], args["search"], args["replace"]
    if search == "":
        return ToolResult(content="empty search block is not allowed", ok=False)
    full = _resolve(ctx.workdir, path)
    if not os.path.isfile(full):
        return ToolResult(content=f"file not found: {path}", ok=False)
    with open(full, encoding="utf-8") as fh:
        text = fh.read()
    count = text.count(search)
    if count == 0:
        return ToolResult(
            content=f"search block not found in {path}; re-read the file and copy an exact snippet",
            ok=False,
        )
    if count > 1:
        return ToolResult(
            content=f"search block matched {count} times in {path}; make it more specific (unique)",
            ok=False,
        )
    new_text = text.replace(search, replace, 1)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    return ToolResult(content=f"edited {path}", ok=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_tools_fs_edit.py -q`
Expected: PASS(5 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools/fs.py packages/code-agent/tests/test_tools_fs_edit.py
git commit -m "feat(code-agent): edit_file SEARCH-REPLACE with uniqueness guardrails"
```

---

## Task 6: shell.bash 工具(超时 + 输出截断)

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/tools/shell.py`
- Test: `packages/code-agent/tests/test_tools_shell.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_tools_shell.py
from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.shell import bash


def test_bash_runs_and_captures_stdout(tmp_path):
    res = bash({"cmd": "echo hello"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "hello" in res.content


def test_bash_runs_in_workdir(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    res = bash({"cmd": "ls"}, ToolContext(workdir=str(tmp_path)))
    assert "marker.txt" in res.content


def test_bash_nonzero_exit_is_not_ok_but_readable(tmp_path):
    res = bash({"cmd": "exit 3"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok is False
    assert "exit code 3" in res.content


def test_bash_timeout(tmp_path):
    res = bash({"cmd": "sleep 5"}, ToolContext(workdir=str(tmp_path), timeout=0.3))
    assert res.ok is False
    assert "timed out" in res.content.lower()


def test_bash_truncates_long_output(tmp_path):
    res = bash(
        {"cmd": "for i in $(seq 1 5000); do echo line$i; done"},
        ToolContext(workdir=str(tmp_path), max_output=300),
    )
    assert res.truncated
    assert "truncated" in res.content.lower()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_tools_shell.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/tools/shell.py
"""bash tool: subprocess with timeout + head/tail output truncation."""

from __future__ import annotations

import subprocess

from anvil_code_agent.tools.base import ToolContext, ToolResult, tool


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return text[:half] + f"\n... [truncated {len(text)} chars] ...\n" + text[-half:], True


@tool(
    name="bash",
    description="Run a shell command in the working dir. Returns combined stdout+stderr.",
    params={"cmd": {"type": "string", "description": "shell command"}},
    required=["cmd"],
)
def bash(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        proc = subprocess.run(
            args["cmd"],
            shell=True,
            cwd=ctx.workdir,
            capture_output=True,
            text=True,
            timeout=ctx.timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(content=f"command timed out after {ctx.timeout}s", ok=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    out, truncated = _truncate(out, ctx.max_output)
    if proc.returncode != 0:
        return ToolResult(content=f"exit code {proc.returncode}\n{out}", ok=False, truncated=truncated)
    return ToolResult(content=out, ok=True, truncated=truncated)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_tools_shell.py -q`
Expected: PASS(5 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools/shell.py packages/code-agent/tests/test_tools_shell.py
git commit -m "feat(code-agent): bash tool with timeout + output truncation"
```

---

## Task 7: verify.run_tests 工具(闭环命脉)

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/tools/verify.py`
- Test: `packages/code-agent/tests/test_tools_verify.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_tools_verify.py
import textwrap

from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.verify import run_tests


def _write(p, name, body):
    (p / name).write_text(textwrap.dedent(body))


def test_run_tests_passing(tmp_path):
    _write(tmp_path, "test_ok.py", "def test_ok():\n    assert 1 + 1 == 2\n")
    res = run_tests({"cmd": "python -m pytest -q"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "passed" in res.content.lower()


def test_run_tests_failing_reports_summary(tmp_path):
    _write(tmp_path, "test_bad.py", "def test_bad():\n    assert 1 + 1 == 3\n")
    res = run_tests({"cmd": "python -m pytest -q"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok is False
    assert "fail" in res.content.lower()


def test_run_tests_default_cmd(tmp_path):
    _write(tmp_path, "test_ok.py", "def test_ok():\n    assert True\n")
    res = run_tests({}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_tools_verify.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/tools/verify.py
"""verify tool: run the project's test command, return structured pass/fail + summary.
This is the closed-loop feedback channel — the agent reads failures and keeps fixing."""

from __future__ import annotations

from anvil_code_agent.tools.base import ToolContext, ToolResult, tool
from anvil_code_agent.tools.shell import bash

DEFAULT_TEST_CMD = "python -m pytest -q"


@tool(
    name="run_tests",
    description="Run the test suite in the working dir. Returns pass/fail and failure output.",
    params={
        "cmd": {"type": "string", "description": f"test command (default: '{DEFAULT_TEST_CMD}')"},
    },
    required=[],
)
def run_tests(args: dict, ctx: ToolContext) -> ToolResult:
    cmd = args.get("cmd") or DEFAULT_TEST_CMD
    res = bash({"cmd": cmd}, ctx)
    # bash already sets ok=False on nonzero exit (pytest exits nonzero on failure);
    # surface the same content but framed as a test verdict.
    verdict = "TESTS PASSED" if res.ok else "TESTS FAILED"
    return ToolResult(content=f"{verdict}\n{res.content}", ok=res.ok, truncated=res.truncated)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_tools_verify.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools/verify.py packages/code-agent/tests/test_tools_verify.py
git commit -m "feat(code-agent): run_tests verify tool (closed-loop feedback)"
```

---

## Task 8: sandbox git worktree 隔离

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/sandbox.py`
- Test: `packages/code-agent/tests/test_sandbox.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_sandbox.py
import os
import subprocess

from anvil_code_agent.sandbox import Worktree


def _init_repo(p):
    subprocess.run(["git", "init", "-q"], cwd=p, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=p, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=p, check=True)
    (p / "f.txt").write_text("orig\n")
    subprocess.run(["git", "add", "."], cwd=p, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=p, check=True)


def test_worktree_creates_isolated_copy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    with Worktree(str(repo)) as wt:
        assert os.path.isfile(os.path.join(wt.path, "f.txt"))
        # 改动隔离:在 worktree 改不影响原仓工作区
        with open(os.path.join(wt.path, "f.txt"), "w") as fh:
            fh.write("changed\n")
        assert (repo / "f.txt").read_text() == "orig\n"


def test_worktree_diff_captures_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    with Worktree(str(repo)) as wt:
        with open(os.path.join(wt.path, "f.txt"), "w") as fh:
            fh.write("changed\n")
        d = wt.diff()
        assert "changed" in d and "orig" in d


def test_worktree_cleanup_removes_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    with Worktree(str(repo)) as wt:
        path = wt.path
    assert not os.path.exists(path)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_sandbox.py -q`
Expected: FAIL(ImportError: Worktree)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/sandbox.py
"""git worktree isolation: each task runs in a throwaway worktree off the target repo.
Changes are isolated, capturable via diff(), and discarded on cleanup."""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid


class Worktree:
    def __init__(self, repo: str):
        self.repo = os.path.abspath(repo)
        self.path = os.path.join(tempfile.gettempdir(), f"anvil-wt-{uuid.uuid4().hex[:8]}")
        self._branch = f"anvil/wt-{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> "Worktree":
        subprocess.run(
            ["git", "worktree", "add", "-q", "-b", self._branch, self.path, "HEAD"],
            cwd=self.repo,
            check=True,
        )
        return self

    def diff(self) -> str:
        return subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def __exit__(self, *exc) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", self.path],
            cwd=self.repo,
            capture_output=True,
        )
        subprocess.run(["git", "branch", "-D", self._branch], cwd=self.repo, capture_output=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_sandbox.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/sandbox.py packages/code-agent/tests/test_sandbox.py
git commit -m "feat(code-agent): git worktree sandbox isolation"
```

---

## Task 9: harness reducer step()(tool_use 往返核心)

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/harness/__init__.py`(空)
- Create: `packages/code-agent/src/anvil_code_agent/harness/loop.py`(本任务先写 `step`)
- Create: `packages/code-agent/tests/conftest.py`(gateway 测试配置)
- Test: `packages/code-agent/tests/test_loop_step.py`

- [ ] **Step 1: 写 conftest + 失败测试**

```python
# packages/code-agent/tests/conftest.py
import os

import pytest


@pytest.fixture(autouse=True)
def _gateway_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k1")
    from anvil_gateway import configure

    configure(
        database_url=os.environ.get(
            "ANVIL_TEST_DATABASE_URL",
            "postgresql+asyncpg://anvil:anvil@localhost:5434/anvil_test",
        ),
        retry_base_delay=0,
    )
```

```python
# packages/code-agent/tests/test_loop_step.py
import json

import httpx
import respx
from anvil_code_agent.harness.loop import step
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _tool_call_resp(name: str, args: dict):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _text_resp(text: str):
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _echo_registry():
    @tool(name="echo", description="d", params={"text": {"type": "string"}}, required=["text"])
    def echo(args, ctx):
        return ToolResult(content="echoed:" + args["text"], ok=True)

    return ToolRegistry([echo])


@respx.mock
async def test_step_executes_tool_call_and_appends_messages(tmp_path):
    respx.post(DS_URL).mock(return_value=_tool_call_resp("echo", {"text": "hi"}))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    s2 = await step(s, "deepseek-chat", _echo_registry(), ToolContext(workdir=str(tmp_path)))
    # 追加了 assistant(带 tool_calls) + tool 结果 两条消息,步进 +1,仍 running
    roles = [m["role"] for m in s2.messages]
    assert roles[-2:] == ["assistant", "tool"]
    assert s2.messages[-1]["tool_call_id"] == "call_1"
    assert "echoed:hi" in s2.messages[-1]["content"]
    assert s2.step == 1 and s2.status == "running"


@respx.mock
async def test_step_finishes_on_text_response(tmp_path):
    respx.post(DS_URL).mock(return_value=_text_resp("all done"))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    s2 = await step(s, "deepseek-chat", _echo_registry(), ToolContext(workdir=str(tmp_path)))
    assert s2.status == "done"
    assert s2.messages[-1]["content"] == "all done"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_loop_step.py -q`
Expected: FAIL(ImportError: step)

- [ ] **Step 3: 实现 step()**

```python
# packages/code-agent/src/anvil_code_agent/harness/__init__.py
```

```python
# packages/code-agent/src/anvil_code_agent/harness/loop.py
"""The reducer loop. step() = one gateway tool_use round-trip + tool execution.
run() drives step() until the model finishes or a guard fires."""

from __future__ import annotations

import json

from anvil_gateway import chat
from anvil_obs import span

from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry


async def step(
    state: AgentState, model: str, registry: ToolRegistry, ctx: ToolContext
) -> AgentState:
    """One reduce: call the model, execute any tool calls, return the new state."""
    resp = await chat(model, list(state.messages), tools=registry.schemas())
    assistant_msg = resp.raw["choices"][0]["message"]
    if resp.tool_calls:
        new = state.append(assistant_msg)
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            with span("code_agent.tool", tool=name):
                result = registry.dispatch(name, args, ctx)
            new = new.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
            )
        return new.advance()
    # no tool calls → the model is done
    return state.append(assistant_msg).advance().finish("done")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_loop_step.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness packages/code-agent/tests/conftest.py packages/code-agent/tests/test_loop_step.py
git commit -m "feat(code-agent): reducer step() — gateway tool_use round-trip"
```

---

## Task 10: harness run() 循环 + 守护

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/harness/loop.py`(追加 `run`)
- Test: `packages/code-agent/tests/test_loop_run.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_loop_run.py
import json

import httpx
import respx
from anvil_code_agent.harness.loop import run
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _tool_call_resp(name, args):
    return httpx.Response(
        200,
        json={
            "id": "x", "model": "deepseek-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]},
                "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def _text_resp(text):
    return httpx.Response(
        200,
        json={"id": "x", "model": "deepseek-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )


def _reg():
    @tool(name="noop", description="d", params={}, required=[])
    def noop(args, ctx):
        return ToolResult(content="ok", ok=True)

    return ToolRegistry([noop])


@respx.mock
async def test_run_loops_until_done(tmp_path):
    # 第一次返回工具调用,第二次返回完成文本
    respx.post(DS_URL).mock(side_effect=[_tool_call_resp("noop", {}), _text_resp("done")])
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=10)
    final = await run(s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)))
    assert final.status == "done"
    assert final.step == 2


@respx.mock
async def test_run_guards_on_max_steps(tmp_path):
    # 永远返回工具调用 → 必须被 max_steps 截断
    respx.post(DS_URL).mock(return_value=_tool_call_resp("noop", {}))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=3)
    final = await run(s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)))
    assert final.status == "exhausted"
    assert final.step == 3
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_loop_run.py -q`
Expected: FAIL(ImportError: run)

- [ ] **Step 3: 实现(追加到 loop.py)**

```python
# 追加到 packages/code-agent/src/anvil_code_agent/harness/loop.py 末尾


async def run(
    state: AgentState, model: str, registry: ToolRegistry, ctx: ToolContext
) -> AgentState:
    """Drive step() until the model finishes or max_steps is hit."""
    with span("code_agent.run", model=model):
        while state.status == "running":
            if state.step >= state.max_steps:
                return state.finish("exhausted")
            state = await step(state, model, registry, ctx)
        return state
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_loop_run.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness/loop.py packages/code-agent/tests/test_loop_run.py
git commit -m "feat(code-agent): run() loop with max_steps guard + obs span"
```

---

## Task 11: eval Task 数据类 + 加载器

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/eval/__init__.py`(空)
- Create: `packages/code-agent/src/anvil_code_agent/eval/task.py`
- Test: `packages/code-agent/tests/test_eval_task.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_eval_task.py
import json

from anvil_code_agent.eval.task import Task, load_tasks


def test_task_fields():
    t = Task(id="t1", repo="fixtures/calc", prompt="fix add", verify_cmd="python -m pytest -q")
    assert t.id == "t1"
    assert t.verify_cmd == "python -m pytest -q"


def test_load_tasks_from_jsonl(tmp_path):
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        json.dumps({"id": "t1", "repo": "fixtures/calc", "prompt": "fix", "verify_cmd": "pytest"})
        + "\n"
    )
    tasks = load_tasks(str(f))
    assert len(tasks) == 1
    assert isinstance(tasks[0], Task)
    assert tasks[0].id == "t1"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_eval_task.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/eval/__init__.py
```

```python
# packages/code-agent/src/anvil_code_agent/eval/task.py
"""Bug-fix eval task: a repo with a failing test the agent must make pass."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    repo: str          # path to the buggy repo (relative to dataset dir or absolute)
    prompt: str        # instruction given to the agent
    verify_cmd: str    # command whose success == task solved


def load_tasks(path: str) -> list[Task]:
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(
                Task(id=d["id"], repo=d["repo"], prompt=d["prompt"], verify_cmd=d["verify_cmd"])
            )
    return tasks
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_eval_task.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/__init__.py packages/code-agent/src/anvil_code_agent/eval/task.py packages/code-agent/tests/test_eval_task.py
git commit -m "feat(code-agent): eval Task dataclass + jsonl loader"
```

---

## Task 12: fixture bug-fix 任务集

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/calc/calc.py`
- Create: `packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/calc/test_calc.py`
- Create: `packages/code-agent/src/anvil_code_agent/eval/golden/tasks.jsonl`
- Test: `packages/code-agent/tests/test_fixture_wellformed.py`

- [ ] **Step 1: 写失败测试(验证 fixture 设计正确)**

```python
# packages/code-agent/tests/test_fixture_wellformed.py
import shutil
import subprocess
from pathlib import Path

from anvil_code_agent.eval.task import load_tasks

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def test_tasks_jsonl_loads():
    tasks = load_tasks(str(GOLDEN / "tasks.jsonl"))
    assert len(tasks) >= 1
    assert any(t.id == "calc-add" for t in tasks)


def test_fixture_test_fails_on_buggy_code(tmp_path):
    # 把 buggy fixture 拷出来,确认它的测试现在就是 FAIL 的(否则任务没意义)
    src = GOLDEN / "fixtures" / "calc"
    dst = tmp_path / "calc"
    shutil.copytree(src, dst)
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True
    )
    assert proc.returncode != 0  # buggy → 测试失败


def test_fixture_passes_after_correct_fix(tmp_path):
    # 应用正确修复后测试应转绿,证明任务可解
    src = GOLDEN / "fixtures" / "calc"
    dst = tmp_path / "calc"
    shutil.copytree(src, dst)
    code = (dst / "calc.py").read_text().replace("a - b", "a + b")
    (dst / "calc.py").write_text(code)
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True
    )
    assert proc.returncode == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_fixture_wellformed.py -q`
Expected: FAIL(文件不存在 / load 失败)

- [ ] **Step 3: 创建 fixture**

```python
# packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/calc/calc.py
def add(a, b):
    return a - b  # BUG: should be a + b


def mul(a, b):
    return a * b
```

```python
# packages/code-agent/src/anvil_code_agent/eval/golden/fixtures/calc/test_calc.py
from calc import add, mul


def test_add():
    assert add(2, 3) == 5


def test_mul():
    assert mul(2, 3) == 6
```

```jsonl
# packages/code-agent/src/anvil_code_agent/eval/golden/tasks.jsonl
{"id": "calc-add", "repo": "fixtures/calc", "prompt": "The test_add test is failing. Read the code, find the bug in calc.py, fix it, and run the tests until they pass.", "verify_cmd": "python -m pytest -q"}
```

(注:jsonl 文件不要写 `#` 注释行,上面注释仅标路径;实际文件只含那一行 JSON。)

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_fixture_wellformed.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/golden packages/code-agent/tests/test_fixture_wellformed.py
git commit -m "feat(code-agent): calc bug-fix fixture + tasks.jsonl (well-formed asserts)"
```

---

## Task 13: eval runner + pass 率

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/eval/runner.py`
- Test: `packages/code-agent/tests/test_eval_runner.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_eval_runner.py
import subprocess
from pathlib import Path

import pytest
from anvil_code_agent.eval.runner import RunResult, solve_task
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_eval_runner.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/eval/runner.py
"""Run an agent against a bug-fix Task in an isolated worktree, then verify with the
task's command. pass = the verify command succeeds after the agent runs."""

from __future__ import annotations

from dataclasses import dataclass

from anvil_code_agent.eval.task import Task
from anvil_code_agent.harness.loop import run
from anvil_code_agent.sandbox import Worktree
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry
from anvil_code_agent.tools.fs import edit_file, read_file
from anvil_code_agent.tools.shell import bash
from anvil_code_agent.tools.verify import run_tests

SYSTEM_PROMPT = (
    "You are a coding agent. Use the tools to read code, edit files, run the tests, "
    "and keep fixing until the tests pass. When the tests pass, stop and say DONE."
)


@dataclass
class RunResult:
    task_id: str
    passed: bool
    steps: int
    diff: str


def default_registry() -> ToolRegistry:
    return ToolRegistry([read_file, edit_file, bash, run_tests])


async def solve_task(task: Task, *, model: str, max_steps: int = 20) -> RunResult:
    with Worktree(task.repo) as wt:
        ctx = ToolContext(workdir=wt.path)
        state = AgentState.new(
            system=SYSTEM_PROMPT, task=task.prompt, workdir=wt.path, max_steps=max_steps
        )
        final = await run(state, model, default_registry(), ctx)
        verdict = run_tests({"cmd": task.verify_cmd}, ctx)
        return RunResult(task_id=task.id, passed=verdict.ok, steps=final.step, diff=wt.diff())


def pass_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.passed) / len(results)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_eval_runner.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/runner.py packages/code-agent/tests/test_eval_runner.py
git commit -m "feat(code-agent): eval runner (worktree → agent → verify) + pass_rate"
```

---

## Task 14: CLI(solve / eval)

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/cli.py`
- Test: `packages/code-agent/tests/test_cli.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_cli.py
from anvil_code_agent.cli import build_parser


def test_parser_has_solve_and_eval():
    p = build_parser()
    ns = p.parse_args(["eval", "--dataset", "tasks.jsonl", "--model", "deepseek-chat"])
    assert ns.command == "eval"
    assert ns.dataset == "tasks.jsonl"
    assert ns.model == "deepseek-chat"


def test_parser_solve():
    p = build_parser()
    ns = p.parse_args(["solve", "--repo", "/tmp/r", "--prompt", "fix it"])
    assert ns.command == "solve"
    assert ns.repo == "/tmp/r"
    assert ns.prompt == "fix it"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_cli.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/cli.py
"""CLI: `anvil-code-agent solve` (one repo+prompt) / `eval` (a tasks.jsonl dataset)."""

from __future__ import annotations

import argparse
import asyncio
import os

from anvil_gateway import configure

from anvil_code_agent.eval.runner import pass_rate, solve_task
from anvil_code_agent.eval.task import Task, load_tasks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anvil-code-agent")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("solve", help="solve one bug-fix task")
    s.add_argument("--repo", required=True)
    s.add_argument("--prompt", required=True)
    s.add_argument("--verify-cmd", default="python -m pytest -q")
    s.add_argument("--model", default="deepseek-chat")
    s.add_argument("--max-steps", type=int, default=20)

    e = sub.add_parser("eval", help="run a tasks.jsonl dataset")
    e.add_argument("--dataset", required=True)
    e.add_argument("--model", default="deepseek-chat")
    e.add_argument("--max-steps", type=int, default=20)
    return p


def _configure_gateway() -> None:
    url = os.environ.get("ANVIL_DATABASE_URL")
    if url:
        configure(database_url=url)


async def _run(ns: argparse.Namespace) -> int:
    _configure_gateway()
    if ns.command == "solve":
        task = Task(id="adhoc", repo=ns.repo, prompt=ns.prompt, verify_cmd=ns.verify_cmd)
        res = await solve_task(task, model=ns.model, max_steps=ns.max_steps)
        print(f"task={res.task_id} passed={res.passed} steps={res.steps}")
        print(res.diff)
        return 0 if res.passed else 1
    # eval
    tasks = load_tasks(ns.dataset)
    results = []
    for t in tasks:
        res = await solve_task(t, model=ns.model, max_steps=ns.max_steps)
        print(f"  {res.task_id}: {'PASS' if res.passed else 'FAIL'} ({res.steps} steps)")
        results.append(res)
    rate = pass_rate(results)
    print(f"pass rate: {rate:.2%} ({sum(r.passed for r in results)}/{len(results)})")
    return 0


def main() -> None:
    ns = build_parser().parse_args()
    raise SystemExit(asyncio.run(_run(ns)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_cli.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/cli.py packages/code-agent/tests/test_cli.py
git commit -m "feat(code-agent): CLI solve/eval subcommands"
```

---

## Task 15: live 冒烟 + 文档接线

**Files:**
- Create: `packages/code-agent/tests/test_live_smoke.py`
- Modify: `anvil/CLAUDE.md`(加 anvil-code-agent 段)
- Create: `examples/06-code-agent/README.md`

- [ ] **Step 1: 写 live 冒烟测试(标记 live,默认不跑)**

```python
# packages/code-agent/tests/test_live_smoke.py
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from anvil_code_agent.eval.runner import solve_task
from anvil_code_agent.eval.task import Task

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def _init_repo(p):
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(cmd, cwd=p, check=True)


@pytest.mark.live
async def test_real_agent_fixes_calc_bug(tmp_path):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("needs DEEPSEEK_API_KEY")
    from anvil_gateway import configure

    configure(database_url=os.environ["ANVIL_DATABASE_URL"])
    repo = tmp_path / "calc"
    shutil.copytree(GOLDEN / "fixtures" / "calc", repo)
    _init_repo(repo)
    from anvil_code_agent.eval.task import load_tasks

    real = load_tasks(str(GOLDEN / "tasks.jsonl"))[0]
    task = Task(id=real.id, repo=str(repo), prompt=real.prompt, verify_cmd=real.verify_cmd)
    res = await solve_task(task, model="deepseek-chat", max_steps=15)
    assert res.passed is True
```

- [ ] **Step 2: 跑测试确认被 skip(无 key)**

Run: `uv run pytest packages/code-agent/tests/test_live_smoke.py -q`
Expected: 1 skipped(无 DEEPSEEK_API_KEY)或在 `-m "not live"` 下 deselect

- [ ] **Step 3: 写文档**

在 `anvil/CLAUDE.md` 末尾追加:

```markdown
## anvil-code-agent (packages/code-agent)

自建编码 agent harness(圈3 agent):reducer 循环 + ACI 工具 + 闭环验证,worktree 隔离内修 bug。

- `harness/loop.py` — `step()` 一次 tool_use 往返;`run()` while 循环 + max_steps 守护
- `tools/` — read_file / edit_file(SEARCH-REPLACE+护栏)/ bash(超时截断)/ run_tests(闭环)
- `sandbox.py` — git worktree 隔离,可 diff 可丢弃
- `eval/` — Task + runner(worktree→agent→verify)+ pass 率;fixture: calc bug-fix
- CLI: `anvil-code-agent solve --repo <r> --prompt "<p>"` / `anvil-code-agent eval --dataset <tasks.jsonl>`
- 测试: `uv run pytest packages/code-agent -q`(工具/沙箱纯本地;loop/runner 走 respx mock + 测试 PG@5434);live 冒烟需 DEEPSEEK_API_KEY
- 复用 gateway(tool_use 往返)/obs(span 追踪每步工具)/eval 基建
```

`examples/06-code-agent/README.md`:

```markdown
# 06 — 自建编码 agent(CA-M1 最小可用循环)

一个 reducer-style while 循环驱动 LLM,用四个工具在 git worktree 隔离内端到端修复 bug。

## 跑 fixture eval(需 DEEPSEEK_API_KEY + ANVIL_DATABASE_URL)

```bash
export ANVIL_DATABASE_URL=postgresql+asyncpg://anvil:anvil@localhost:5434/anvil
uv run anvil-code-agent eval --dataset packages/code-agent/src/anvil_code_agent/eval/golden/tasks.jsonl
```

预期:agent 读 calc.py → 发现 `a - b` 应为 `a + b` → edit_file 修正 → run_tests 转绿 → PASS。

## 设计要点

- **agent 即 reducer**:`step(state) -> state'`,纯函数式,易测易恢复
- **编辑可靠性**:SEARCH-REPLACE 唯一匹配护栏,不匹配就报错让模型重读重试(决定可用性)
- **闭环**:run_tests 的失败摘要喂回 context,模型据此继续改
- **隔离**:worktree per task,改动可整体 diff、可丢弃
```

- [ ] **Step 4: 跑全量测试 + lint**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest packages/code-agent -m "not live" -q && uv run ruff check packages/code-agent`
Expected: 全绿(约 30+ passed,1 deselected/skipped)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/tests/test_live_smoke.py CLAUDE.md examples/06-code-agent
git commit -m "feat(code-agent): live smoke test + CLAUDE/example docs (CA-M1 complete)"
```

---

## 完成标准(CA-M1 验收)

- `uv run pytest packages/code-agent -m "not live" -q` 全绿。
- `uv run ruff check packages/code-agent` 无错。
- live 冒烟(配 key)能让真实模型端到端把 calc bug 修绿、`solve_task.passed is True`。
- CLI `anvil-code-agent eval` 打印 pass 率。
- 触底:agent loop + tool_use 协议全往返 + SEARCH-REPLACE 编辑护栏 + run_tests 闭环反馈。
