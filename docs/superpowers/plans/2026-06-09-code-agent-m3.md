# CA-M3 工程化(context 压缩 + 权限门 + Docker 沙箱 + 断点恢复)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** 把 M1 的最小循环工程化为可长跑、可控、可隔离、可恢复的 harness:token 预算下的上下文压缩、工具风险分级审批门、Docker 容器沙箱(可选后端)、AgentState 序列化断点恢复。

**Architecture:** 四个独立单元,各自接进现有 reducer 循环而不破坏 M1/M2:`context.py`(估 token + 结构安全压缩,发送前应用)、`permission.py`(风险分级 + 审批策略回调,dispatch 前拦截)、`sandbox.py`(给 `ToolContext` 加 `executor` 接缝 + `DockerSandbox` 用 docker CLI 在容器内执行)、`recovery.py`(state 序列化/反序列化,可 dump→load→resume)。`step()`/`run()` 加 `policy`/`token_budget` 可选参,默认值保持 M1 行为不变。

**Tech Stack:** 纯 Python;Docker 经 `subprocess` 调 docker CLI(零新依赖);压缩用字符启发式估 token。

---

## 文件结构

- Create: `packages/code-agent/src/anvil_code_agent/harness/context.py` — estimate_tokens + compact
- Create: `packages/code-agent/src/anvil_code_agent/harness/permission.py` — risk_level + 审批策略
- Create: `packages/code-agent/src/anvil_code_agent/harness/recovery.py` — dump_state/load_state
- Modify: `packages/code-agent/src/anvil_code_agent/tools/base.py` — ToolContext 加 `executor`
- Modify: `packages/code-agent/src/anvil_code_agent/tools/shell.py` — bash 尊重 ctx.executor
- Modify: `packages/code-agent/src/anvil_code_agent/sandbox.py` — 追加 DockerSandbox
- Modify: `packages/code-agent/src/anvil_code_agent/harness/loop.py` — step/run 加 policy+token_budget
- Modify: `CLAUDE.md` + `examples/06-code-agent/README.md`

---

## Task 1: context.py — 估 token + 结构安全压缩

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/harness/context.py`
- Test: `packages/code-agent/tests/test_context.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_context.py
from anvil_code_agent.harness.context import compact, estimate_tokens


def _msgs():
    return [
        {"role": "system", "content": "you are a coder"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "X" * 4000},  # 老的大工具输出
        {"role": "assistant", "content": "still working"},
        {"role": "user", "content": "keep going"},
    ]


def test_estimate_tokens_grows_with_content():
    small = [{"role": "user", "content": "hi"}]
    big = [{"role": "user", "content": "hi" * 1000}]
    assert estimate_tokens(big) > estimate_tokens(small)
    assert estimate_tokens([]) == 0


def test_compact_noop_under_budget():
    m = _msgs()
    assert compact(m, max_tokens=10_000) == m  # 未超预算原样返回


def test_compact_truncates_old_tool_output_but_keeps_structure():
    m = _msgs()
    out = compact(m, max_tokens=100, keep_recent=2, tool_cap=120)
    # 消息条数、role 序列、tool_call_id 全保留(不破坏 tool_use 配对)
    assert len(out) == len(m)
    assert [x["role"] for x in out] == [x["role"] for x in m]
    assert out[3]["tool_call_id"] == "c1"
    # 老的大工具输出被截断 → 总 token 下降
    assert estimate_tokens(out) < estimate_tokens(m)
    assert "truncated" in out[3]["content"]


def test_compact_protects_system_task_and_recent():
    m = _msgs()
    out = compact(m, max_tokens=1, keep_recent=2, tool_cap=50)
    assert out[0] == m[0]  # system 不动
    assert out[1] == m[1]  # 原始任务不动
    assert out[-1] == m[-1]  # 最近窗口不动
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest packages/code-agent/tests/test_context.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/harness/context.py
"""Context engineering: estimate token budget + structurally-safe compaction.

Compaction only TRUNCATES the content of old tool-result messages — it never drops or
reorders messages, so tool_call/tool pairing stays valid. This is best-effort: if the
recent window itself exceeds budget it may not fully fit, which is the honest behavior.
(LLM summarization of dropped turns is a future tier.)"""

from __future__ import annotations

from typing import Any

Message = dict[str, Any]


def estimate_tokens(messages: list[Message]) -> int:
    """Cheap heuristic: ~4 chars per token over all stringified content/tool_calls."""
    chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        if m.get("tool_calls"):
            chars += len(str(m["tool_calls"]))
    return chars // 4


def compact(
    messages: list[Message],
    *,
    max_tokens: int,
    keep_recent: int = 6,
    tool_cap: int = 200,
) -> list[Message]:
    """Return messages shrunk toward max_tokens by truncating old tool outputs.
    Protects messages[0] (system), messages[1] (original task), and the last keep_recent."""
    if estimate_tokens(messages) <= max_tokens:
        return messages
    n = len(messages)
    out: list[Message] = []
    for i, m in enumerate(messages):
        protected = i < 2 or i >= n - keep_recent
        content = m.get("content")
        if (
            not protected
            and m.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > tool_cap
        ):
            mm = dict(m)
            mm["content"] = content[:tool_cap] + " ...[older tool output truncated]"
            out.append(mm)
        else:
            out.append(m)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_context.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness/context.py packages/code-agent/tests/test_context.py
git commit -m "feat(code-agent): context.py — token estimate + structurally-safe compaction"
```

---

## Task 2: permission.py — 风险分级 + 审批策略

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/harness/permission.py`
- Test: `packages/code-agent/tests/test_permission.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_permission.py
from anvil_code_agent.harness.permission import (
    auto_approve,
    deny_high,
    gate_by_risk,
    risk_level,
)


def test_risk_levels():
    assert risk_level("read_file") == "low"
    assert risk_level("grep") == "low"
    assert risk_level("edit_file") == "medium"
    assert risk_level("bash") == "high"
    assert risk_level("unknown_tool") == "high"  # 未知工具按最高风险(安全默认)


def test_auto_approve_allows_all():
    assert auto_approve("bash", {"cmd": "rm -rf /"}, "high") is True


def test_deny_high_blocks_only_high():
    assert deny_high("read_file", {}, "low") is True
    assert deny_high("edit_file", {}, "medium") is True
    assert deny_high("bash", {}, "high") is False


def test_gate_by_risk_factory():
    only_low = gate_by_risk("low")  # 只放行 <= low
    assert only_low("read_file", {}, "low") is True
    assert only_low("edit_file", {}, "medium") is False
    up_to_medium = gate_by_risk("medium")
    assert up_to_medium("edit_file", {}, "medium") is True
    assert up_to_medium("bash", {}, "high") is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_permission.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/harness/permission.py
"""Tool risk tiers + approval policies. A policy is a callable
(name, args, risk) -> bool; returning False blocks the tool (the loop turns the block
into model feedback). Default policy in eval is auto_approve; interactive callers can
swap in a gate that requires human approval for high-risk tools."""

from __future__ import annotations

from typing import Any, Callable

ApprovalPolicy = Callable[[str, dict[str, Any], str], bool]

_RISK = {
    "read_file": "low",
    "grep": "low",
    "repo_map": "low",
    "edit_file": "medium",
    "run_tests": "medium",
    "bash": "high",
}
_ORDER = {"low": 0, "medium": 1, "high": 2}


def risk_level(name: str) -> str:
    """Unknown tools default to 'high' — fail safe."""
    return _RISK.get(name, "high")


def auto_approve(name: str, args: dict[str, Any], risk: str) -> bool:
    return True


def deny_high(name: str, args: dict[str, Any], risk: str) -> bool:
    return risk != "high"


def gate_by_risk(max_risk: str) -> ApprovalPolicy:
    """Allow tools whose risk is <= max_risk; block the rest."""
    ceiling = _ORDER[max_risk]

    def policy(name: str, args: dict[str, Any], risk: str) -> bool:
        return _ORDER.get(risk, 2) <= ceiling

    return policy
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_permission.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness/permission.py packages/code-agent/tests/test_permission.py
git commit -m "feat(code-agent): permission.py — tool risk tiers + approval policies"
```

---

## Task 3: 把 compact + policy 接进 step/run

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/harness/loop.py`
- Test: `packages/code-agent/tests/test_loop_m3.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_loop_m3.py
import json

import httpx
import respx
from anvil_code_agent.harness.loop import run, step
from anvil_code_agent.harness.permission import deny_high
from anvil_code_agent.state import AgentState
from anvil_code_agent.tools.base import ToolContext, ToolRegistry, ToolResult, tool

DS_URL = "https://api.deepseek.com/v1/chat/completions"


def _bash_call(cmd):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"cmd": cmd})}}]},
         "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _text(t):
    return httpx.Response(200, json={"id": "x", "model": "deepseek-chat", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": t}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _reg():
    @tool(name="bash", description="d", params={"cmd": {"type": "string"}}, required=["cmd"])
    def bash(args, ctx):
        return ToolResult(content="ran:" + args["cmd"], ok=True)

    return ToolRegistry([bash])


@respx.mock
async def test_policy_blocks_high_risk_tool_as_feedback(tmp_path):
    respx.post(DS_URL).mock(return_value=_bash_call("echo hi"))
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    # bash 是 high 风险;deny_high 应拦截 → 工具结果是"denied"反馈,不执行
    s2 = await step(s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)), policy=deny_high)
    tool_msg = s2.messages[-1]
    assert tool_msg["role"] == "tool"
    assert "denied" in tool_msg["content"].lower()
    assert "ran:" not in tool_msg["content"]  # 没真执行


@respx.mock
async def test_run_passes_policy_and_budget_through(tmp_path):
    respx.post(DS_URL).mock(side_effect=[_bash_call("ls"), _text("done")])
    s = AgentState.new(system="s", task="t", workdir=str(tmp_path), max_steps=5)
    final = await run(s, "deepseek-chat", _reg(), ToolContext(workdir=str(tmp_path)),
                      policy=deny_high, token_budget=10_000)
    assert final.status == "done"
    # bash 被拦,但循环没崩,继续到 done
    assert any("denied" in m.get("content", "") for m in final.messages if m["role"] == "tool")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_loop_m3.py -q`
Expected: FAIL(step/run 不接受 policy/token_budget)

- [ ] **Step 3: 实现(改 loop.py)**

把 `loop.py` 顶部 import 区补:

```python
from anvil_code_agent.harness.context import compact
from anvil_code_agent.harness.permission import ApprovalPolicy, auto_approve, risk_level
```

把 `step` 改为(新增两个 keyword-only 参数,逻辑插入 compact 与 policy 拦截):

```python
async def step(
    state: AgentState,
    model: str,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    policy: ApprovalPolicy = auto_approve,
    token_budget: int | None = None,
) -> AgentState:
    """One reduce: call the model, execute any tool calls, return the new state."""
    msgs = list(state.messages)
    if token_budget is not None:
        msgs = compact(msgs, max_tokens=token_budget)
    resp = await chat(model, msgs, tools=registry.schemas())
    assistant_msg = resp.raw["choices"][0]["message"]
    if resp.tool_calls:
        new = state.append(assistant_msg)
        for tc in resp.tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            risk = risk_level(name)
            if not policy(name, args, risk):
                new = new.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": f"tool '{name}' denied by approval policy (risk={risk})",
                    }
                )
                continue
            with span("code_agent.tool", tool=name, risk=risk):
                result = registry.dispatch(name, args, ctx)
            new = new.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
            )
        return new.advance()
    return state.append(assistant_msg).advance().finish("done")
```

把 `run` 改为透传两个参数:

```python
async def run(
    state: AgentState,
    model: str,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    policy: ApprovalPolicy = auto_approve,
    token_budget: int | None = None,
) -> AgentState:
    """Drive step() until the model finishes or max_steps is hit."""
    with span("code_agent.run", model=model):
        while state.status == "running":
            if state.step >= state.max_steps:
                return state.finish("exhausted")
            state = await step(
                state, model, registry, ctx, policy=policy, token_budget=token_budget
            )
        return state
```

- [ ] **Step 4: 跑测试确认通过 + 回归 M1 loop 测试**

Run: `uv run pytest packages/code-agent/tests/test_loop_m3.py packages/code-agent/tests/test_loop_step.py packages/code-agent/tests/test_loop_run.py -q`
Expected: 全 PASS(M1 默认行为不变 + M3 新行为)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness/loop.py packages/code-agent/tests/test_loop_m3.py
git commit -m "feat(code-agent): wire compaction + approval policy into step/run (defaults preserve M1)"
```

---

## Task 4: ToolContext.executor 接缝 + bash 尊重它

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/tools/base.py`(ToolContext 加 executor)
- Modify: `packages/code-agent/src/anvil_code_agent/tools/shell.py`(bash 走 executor)
- Test: `packages/code-agent/tests/test_executor_seam.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_executor_seam.py
from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.shell import bash


def test_bash_uses_executor_when_present(tmp_path):
    seen = {}

    def fake_executor(cmd):
        seen["cmd"] = cmd
        return (0, "from-executor")

    ctx = ToolContext(workdir=str(tmp_path), executor=fake_executor)
    res = bash({"cmd": "echo hi"}, ctx)
    assert res.ok
    assert res.content == "from-executor"
    assert seen["cmd"] == "echo hi"  # 路由到 executor,没走 host subprocess


def test_bash_executor_nonzero_is_error(tmp_path):
    ctx = ToolContext(workdir=str(tmp_path), executor=lambda c: (2, "boom"))
    res = bash({"cmd": "false"}, ctx)
    assert res.ok is False
    assert "exit code 2" in res.content
    assert "boom" in res.content


def test_bash_falls_back_to_host_without_executor(tmp_path):
    res = bash({"cmd": "echo host"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "host" in res.content
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_executor_seam.py -q`
Expected: FAIL(ToolContext 无 executor 参数)

- [ ] **Step 3: 实现**

`tools/base.py` 把 `ToolContext` 改为(加 `executor` 字段):

```python
@dataclass
class ToolContext:
    workdir: str
    timeout: float = 120.0
    max_output: int = 4096
    executor: Callable[[str], tuple[int, str]] | None = None
```

(确保 `base.py` 顶部已 `from typing import Any, Callable`——M1 已导入 Callable。)

`tools/shell.py` 的 `bash` 改为(开头加 executor 分支,host 路径保留):

```python
@tool(
    name="bash",
    description="Run a shell command in the working dir. Returns combined stdout+stderr. "
    "Paths/commands are not sandboxed in M1 unless a Docker executor is attached (M3).",
    params={"cmd": {"type": "string", "description": "shell command"}},
    required=["cmd"],
)
def bash(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.executor is not None:
        rc, raw = ctx.executor(args["cmd"])
        out, truncated = _truncate(raw, ctx.max_output)
        if rc != 0:
            return ToolResult(content=f"exit code {rc}\n{out}", ok=False, truncated=truncated)
        return ToolResult(content=out, ok=True, truncated=truncated)
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

- [ ] **Step 4: 跑测试确认通过 + 回归 M1 shell/verify 测试**

Run: `uv run pytest packages/code-agent/tests/test_executor_seam.py packages/code-agent/tests/test_tools_shell.py packages/code-agent/tests/test_tools_verify.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools/base.py packages/code-agent/src/anvil_code_agent/tools/shell.py packages/code-agent/tests/test_executor_seam.py
git commit -m "feat(code-agent): ToolContext.executor seam — bash routes to sandbox executor"
```

---

## Task 5: DockerSandbox(docker CLI,容器内执行)

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/sandbox.py`(追加 DockerSandbox + has_docker)
- Test: `packages/code-agent/tests/test_docker_sandbox.py`

- [ ] **Step 1: 写失败测试(无 docker 自动跳过)**

```python
# packages/code-agent/tests/test_docker_sandbox.py
import pytest
from anvil_code_agent.sandbox import DockerSandbox, has_docker

pytestmark = pytest.mark.skipif(not has_docker(), reason="docker daemon unavailable")


def test_docker_executor_runs_in_container(tmp_path):
    (tmp_path / "hello.txt").write_text("hi\n")
    with DockerSandbox(str(tmp_path)) as box:
        rc, out = box.exec("cat hello.txt")
        assert rc == 0
        assert "hi" in out
        # 容器隔离:容器里能跑命令
        rc2, out2 = box.exec("python -c 'print(2+2)'")
        assert rc2 == 0 and "4" in out2


def test_docker_executor_nonzero_exit(tmp_path):
    with DockerSandbox(str(tmp_path)) as box:
        rc, out = box.exec("exit 7")
        assert rc == 7


def test_docker_sandbox_cleans_up_container(tmp_path):
    import subprocess

    with DockerSandbox(str(tmp_path)) as box:
        name = box.name
    # 退出后容器应已删除
    res = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"],
                         capture_output=True, text=True)
    assert res.stdout.strip() == ""
```

- [ ] **Step 2: 跑测试确认失败/跳过**

Run: `uv run pytest packages/code-agent/tests/test_docker_sandbox.py -q`
Expected: FAIL(ImportError: DockerSandbox)——有 docker 时;无 docker 则 skip

- [ ] **Step 3: 实现(追加到 sandbox.py)**

```python
# 追加到 packages/code-agent/src/anvil_code_agent/sandbox.py 末尾


def has_docker() -> bool:
    """True if a docker daemon is reachable."""
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerSandbox:
    """Process-isolated sandbox: a container with the work dir bind-mounted at /work.
    Edit files on the host path; run commands inside the container via exec()."""

    def __init__(self, workdir: str, image: str = "python:3.12-slim"):
        self.workdir = os.path.abspath(workdir)
        self.image = image
        self.name = f"anvil-box-{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> "DockerSandbox":
        subprocess.run(
            [
                "docker", "run", "-d", "--name", self.name,
                "-v", f"{self.workdir}:/work", "-w", "/work",
                self.image, "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )
        return self

    def exec(self, cmd: str, timeout: float = 120.0) -> tuple[int, str]:
        try:
            r = subprocess.run(
                ["docker", "exec", "-w", "/work", self.name, "sh", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return (124, f"command timed out after {timeout}s")
        return (r.returncode, (r.stdout or "") + (r.stderr or ""))

    def __exit__(self, *exc) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
```

(注:`sandbox.py` 顶部已 `import os/subprocess/uuid`——M1 已有。)

- [ ] **Step 4: 跑测试确认通过(本机有 docker)**

Run: `uv run pytest packages/code-agent/tests/test_docker_sandbox.py -q`
Expected: PASS(3 passed)——本机 docker 可用;CI 无 docker 则 skip

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/sandbox.py packages/code-agent/tests/test_docker_sandbox.py
git commit -m "feat(code-agent): DockerSandbox — container-isolated exec via docker CLI"
```

---

## Task 6: recovery.py — 序列化断点恢复

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/harness/recovery.py`
- Test: `packages/code-agent/tests/test_recovery.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_recovery.py
import json

from anvil_code_agent.harness.recovery import dump_state, load_state
from anvil_code_agent.state import AgentState


def test_dump_load_roundtrip():
    s = AgentState.new(system="s", task="t", workdir="/tmp/wt", max_steps=10)
    s = s.append({"role": "assistant", "content": "hi"}).advance()
    d = dump_state(s)
    # 必须 JSON 可序列化(断点落盘)
    blob = json.dumps(d)
    s2 = load_state(json.loads(blob))
    assert isinstance(s2, AgentState)
    assert s2.messages == s.messages
    assert s2.step == s.step
    assert s2.max_steps == s.max_steps
    assert s2.workdir == s.workdir
    assert s2.status == s.status


def test_loaded_state_is_resumable():
    s = AgentState.new(system="s", task="t", workdir="/tmp", max_steps=5)
    s = s.append({"role": "user", "content": "more"}).advance().advance()
    s2 = load_state(dump_state(s))
    # 还能继续推进(reducer 语义完好)
    s3 = s2.advance()
    assert s3.step == s.step + 1
    assert s3.status == "running"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_recovery.py -q`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/harness/recovery.py
"""Checkpoint recovery: dump an AgentState to a JSON-safe dict and reload it.
messages are already plain dicts, so persistence is trivial — enabling resume after
a crash or a deliberate pause (12-Factor #6: launch/pause/resume)."""

from __future__ import annotations

from typing import Any

from anvil_code_agent.state import AgentState


def dump_state(state: AgentState) -> dict[str, Any]:
    return {
        "messages": [dict(m) for m in state.messages],
        "step": state.step,
        "max_steps": state.max_steps,
        "workdir": state.workdir,
        "status": state.status,
    }


def load_state(d: dict[str, Any]) -> AgentState:
    return AgentState(
        messages=tuple(d["messages"]),
        step=d["step"],
        max_steps=d["max_steps"],
        workdir=d["workdir"],
        status=d["status"],
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_recovery.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness/recovery.py packages/code-agent/tests/test_recovery.py
git commit -m "feat(code-agent): recovery.py — JSON dump/load AgentState for resume"
```

---

## Task 7: 把 DockerSandbox 接进 runner(opt-in,让能力可达)

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/eval/runner.py`(solve_task 加 `use_docker`)
- Modify: `packages/code-agent/src/anvil_code_agent/cli.py`(solve/eval 加 `--docker` 开关)
- Test: `packages/code-agent/tests/test_runner_docker.py`

- [ ] **Step 1: 写失败测试(无 docker 跳过真容器;always 测默认不开)**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_runner_docker.py -q`
Expected: FAIL(solve_task 无 use_docker)

- [ ] **Step 3: 实现(改 runner.py)**

`runner.py` 顶部 import 区追加:

```python
from anvil_code_agent.sandbox import DockerSandbox, Worktree, has_docker
```

(把原来的 `from anvil_code_agent.sandbox import Worktree` 替换为上面这行。)

把 `solve_task` 改为(加 `use_docker`,docker 模式下用 DockerSandbox 接 executor + 容器内 verify):

```python
async def solve_task(
    task: Task, *, model: str, max_steps: int = 20, use_docker: bool = False
) -> RunResult:
    with Worktree(task.repo) as wt:
        if use_docker and has_docker():
            with DockerSandbox(wt.path) as box:
                ctx = ToolContext(workdir=wt.path, executor=box.exec)
                state = AgentState.new(
                    system=SYSTEM_PROMPT, task=task.prompt, workdir=wt.path, max_steps=max_steps
                )
                final = await run(state, model, default_registry(), ctx)
                rc, out = box.exec(task.verify_cmd)
                return RunResult(task_id=task.id, passed=rc == 0, steps=final.step, diff=wt.diff())
        ctx = ToolContext(workdir=wt.path)
        state = AgentState.new(
            system=SYSTEM_PROMPT, task=task.prompt, workdir=wt.path, max_steps=max_steps
        )
        final = await run(state, model, default_registry(), ctx)
        verdict = run_tests({"cmd": task.verify_cmd}, ctx)
        return RunResult(task_id=task.id, passed=verdict.ok, steps=final.step, diff=wt.diff())
```

`cli.py`:给 `solve` 与 `eval` 子命令各加 `--docker` 开关,并把它传进 `solve_task`。

在 `build_parser` 的 solve 解析器加:

```python
    s.add_argument("--docker", action="store_true", help="run commands in a Docker sandbox")
```

eval 解析器加:

```python
    e.add_argument("--docker", action="store_true", help="run commands in a Docker sandbox")
```

`_run` 里两处 `solve_task(...)` 调用都加 `use_docker=ns.docker`:

```python
        res = await solve_task(task, model=ns.model, max_steps=ns.max_steps, use_docker=ns.docker)
```
```python
        res = await solve_task(t, model=ns.model, max_steps=ns.max_steps, use_docker=ns.docker)
```

- [ ] **Step 4: 跑测试确认通过 + 回归 runner/cli 测试**

Run: `uv run pytest packages/code-agent/tests/test_runner_docker.py packages/code-agent/tests/test_eval_runner.py packages/code-agent/tests/test_cli.py packages/code-agent/tests/test_cli_eval.py -q`
Expected: 全 PASS(docker 真容器测试本机跑、CI skip)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/eval/runner.py packages/code-agent/src/anvil_code_agent/cli.py packages/code-agent/tests/test_runner_docker.py
git commit -m "feat(code-agent): runner/CLI --docker opt-in (wire DockerSandbox executor)"
```

---

## Task 8: 文档接线 + 全量回归

**Files:**
- Modify: `anvil/CLAUDE.md`、`examples/06-code-agent/README.md`

- [ ] **Step 1: 全量回归(根目录,查 collision)**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest -m "not live" -q && uv run ruff check packages/code-agent`
Expected: 全绿(docker 测试本机跑、CI skip),ruff 净

- [ ] **Step 2: 写文档**

`anvil/CLAUDE.md` 的 anvil-code-agent 段落补:

```markdown
- `harness/context.py`(M3)— 字符估 token + 结构安全压缩(只截老工具输出,不破坏 tool_use 配对),发送前按 token_budget 应用
- `harness/permission.py`(M3)— 工具风险分级(read/grep/repo_map=low,edit/run_tests=medium,bash=high,未知=high)+ 审批策略回调(auto_approve/deny_high/gate_by_risk),dispatch 前拦截,拦截转模型反馈
- `harness/recovery.py`(M3)— AgentState ↔ JSON dump/load,断点落盘可 resume
- `sandbox.py` DockerSandbox(M3)— docker CLI 起容器,workdir bind-mount 到 /work,`exec()` 容器内执行;`ToolContext.executor` 接缝让 bash 路由进容器(host 为默认回退)
- step/run 加 `policy`/`token_budget` 可选参,默认保持 M1 行为
```

`examples/06-code-agent/README.md` 末尾追加:

```markdown
## CA-M3:工程化(可长跑 / 可控 / 可隔离 / 可恢复)

- **上下文压缩**:`estimate_tokens` + `compact` ——超预算时截断老的大工具输出,但绝不删消息/打乱顺序(保住 tool_use 配对的结构正确)。发送前应用,完整历史仍留存供 diff/恢复。
- **权限门**:工具按风险分三档(low/medium/high,未知按 high 安全默认),审批策略是 `(name,args,risk)->bool` 回调;eval 用 auto_approve,交互可换成对 high 风险要人批的门。被拦的工具不执行、转成模型反馈。
- **Docker 沙箱**:`DockerSandbox` 用 docker CLI 起容器、bind-mount workdir,bash 经 `ToolContext.executor` 路由进容器执行——跑任意 shell 才真正进程级隔离(host 子进程是默认回退)。
- **断点恢复**:AgentState 全是普通 dict,`dump_state`/`load_state` 一行落盘/重载,崩溃或主动暂停后可 resume(12-Factor #6)。
```

- [ ] **Step 3: 提交**

```bash
git add CLAUDE.md examples/06-code-agent/README.md
git commit -m "docs(code-agent): CA-M3 context/permission/docker/recovery (milestone complete)"
```

---

## 完成标准(CA-M3 验收)

- `uv run pytest packages/code-agent -m "not live" -q` 全绿(docker 测试本机跑、CI skip);ruff 净;根目录全量无 collision。
- compact 超预算截老工具输出且保结构;policy 拦 high 风险转反馈;DockerSandbox 容器内执行并清理;state dump→load→resume 完好。
- step/run 默认参数下 M1/M2 行为零变化(回归绿)。
- 触底:上下文工程(token 预算+压缩)、工具风险审批、容器隔离、interrupt/resume 持久化。
