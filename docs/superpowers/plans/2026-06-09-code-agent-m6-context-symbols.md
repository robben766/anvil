# CA-M6 上下文摘要压缩 + repo map 符号级排名 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** 两项独立的螺旋加深——(A) 给 context 压缩加 **LLM 摘要 tier**(超预算时把更老的回合摘成一条,而非只截断),(B) repo map 从"文件级"细化到 **符号级排名**(按符号被引用次数排,枢纽符号优先 + 每文件 top-K)。

**Architecture:** (A) `context.compact` 增可选 `summarizer` 回调:截断后仍超预算时,把"中间区"(系统/任务/最近窗口之外)整段替换为一条摘要消息,**替换边界对齐到非 tool 消息**以不破坏 tool_use 配对;`llm_summarizer(model)` 提供 gateway 实现。(B) `build_repo_map` 计算每个 def 符号的全仓被引次数,渲染每个文件时按被引次数降序取 top-K。两项默认行为温和(summarizer 默认 None;符号排名替换原字母序但仍列全符号到 top-K 上限)。

**Tech Stack:** 纯 Python;摘要 tier 的 LLM 实现走 anvil-gateway(测试用 fake summarizer)。

---

## 文件结构

- Modify: `packages/code-agent/src/anvil_code_agent/harness/context.py` — compact += summarizer + llm_summarizer
- Modify: `packages/code-agent/src/anvil_code_agent/harness/loop.py` — step/run += summarizer
- Modify: `packages/code-agent/src/anvil_code_agent/repomap/build.py` — 符号级排名
- Modify: `CLAUDE.md` + `examples/06-code-agent/README.md`

---

## Task 1: compact 增 LLM 摘要 tier(回合边界安全)

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/harness/context.py`
- Test: `packages/code-agent/tests/test_context_summary.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_context_summary.py
from anvil_code_agent.harness.context import compact, estimate_tokens


def _convo():
    # system, task, 然后多个"回合":assistant(tool_calls)+tool ... 最后 user
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(6):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * 500})
    msgs.append({"role": "user", "content": "recent ask"})
    return msgs


def test_summarizer_replaces_middle_and_preserves_pairing():
    m = _convo()
    summary_calls = {}

    def fake_summarizer(middle):
        summary_calls["n"] = len(middle)
        return "did some reading"

    out = compact(m, max_tokens=50, keep_recent=3, tool_cap=80, summarizer=fake_summarizer)
    # 摘要被调用、生成了一条摘要消息
    assert summary_calls.get("n", 0) > 0
    assert any("did some reading" in (x.get("content") or "") for x in out)
    # system + task 仍在最前
    assert out[0] == m[0] and out[1] == m[1]
    # 最近窗口仍在最后
    assert out[-1] == m[-1]
    # tool_use 配对完好:每个 role==tool 的前面都有带该 tool_call_id 的 assistant
    open_ids = set()
    for x in out:
        if x["role"] == "assistant" and x.get("tool_calls"):
            for tc in x["tool_calls"]:
                open_ids.add(tc["id"])
        if x["role"] == "tool":
            assert x["tool_call_id"] in open_ids  # 不孤儿
    # 总 token 下降
    assert estimate_tokens(out) < estimate_tokens(m)


def test_no_summarizer_falls_back_to_truncation_only():
    m = _convo()
    out = compact(m, max_tokens=50, keep_recent=3, tool_cap=80)  # 无 summarizer
    # 行为同 M3:条数不变(只截断,不替换)
    assert len(out) == len(m)


def test_summarizer_not_called_when_under_budget():
    m = _convo()
    called = {"v": False}

    def fake(middle):
        called["v"] = True
        return "x"

    out = compact(m, max_tokens=10_000, keep_recent=3, summarizer=fake)
    assert called["v"] is False
    assert out == m
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest packages/code-agent/tests/test_context_summary.py -q`
Expected: FAIL(compact 不接受 summarizer)

- [ ] **Step 3: 实现(改 context.py)**

把 `context.py` 顶部 import 补 `Callable`:

```python
from collections.abc import Callable
```

把 `compact` 改为(在原截断逻辑后追加摘要 tier):

```python
def compact(
    messages: list[Message],
    *,
    max_tokens: int,
    keep_recent: int = 6,
    tool_cap: int = 200,
    summarizer: Callable[[list[Message]], str] | None = None,
) -> list[Message]:
    """Shrink messages toward max_tokens. Tier 1: truncate old tool-message content.
    Tier 2 (if summarizer given and still over budget): replace the middle region
    (everything after system+task, before the recent window) with a single summary
    message. The cut is aligned to a non-tool boundary so tool_call/tool pairing stays
    valid. Protects messages[0] (system), messages[1] (task), and the last keep_recent."""
    if estimate_tokens(messages) <= max_tokens:
        return messages
    n = len(messages)
    # Tier 1: truncate old tool outputs (structurally safe).
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
    # Tier 2: summarize the middle if still over budget and a summarizer is available.
    if summarizer is None or estimate_tokens(out) <= max_tokens:
        return out
    cut = n - keep_recent
    # back the cut up to a non-tool message so the kept tail starts cleanly (no orphan tool)
    while cut > 2 and out[cut].get("role") == "tool":
        cut -= 1
    if cut <= 2:
        return out  # nothing summarizable without breaking pairing
    middle = out[2:cut]
    summary = summarizer(middle)
    summary_msg = {"role": "user", "content": f"[Summary of earlier work]\n{summary}"}
    return out[:2] + [summary_msg] + out[cut:]
```

追加一个 gateway 实现的摘要器工厂(放到 context.py 末尾):

```python
def llm_summarizer(model: str) -> Callable[[list[Message]], str]:
    """A summarizer backed by the gateway. compact() is sync but is called from inside the
    async agent loop, so we run the async chat() in a fresh event loop on a worker thread
    (calling run_until_complete on the already-running loop would raise)."""
    import asyncio
    import concurrent.futures

    from anvil_gateway import chat

    def summarize(middle: list[Message]) -> str:
        transcript = "\n".join(
            f"{m.get('role')}: {(m.get('content') or '')[:500]}" for m in middle
        )
        msgs = [
            {"role": "system", "content": "Summarize the assistant's earlier work concisely "
             "(files read, edits made, test results) in 3-5 bullet points."},
            {"role": "user", "content": transcript},
        ]

        def _call() -> str:
            resp = asyncio.run(chat(model, msgs))
            return resp.content or "(no summary)"

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_call).result()

    return summarize
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_context_summary.py packages/code-agent/tests/test_context.py -q`
Expected: 全 PASS(新摘要 tier + M3 截断行为不变)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness/context.py packages/code-agent/tests/test_context_summary.py
git commit -m "feat(code-agent): context summary tier — replace middle turns with LLM summary"
```

---

## Task 2: 把 summarizer 接进 step/run

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/harness/loop.py`
- Test: `packages/code-agent/tests/test_loop_summary.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_loop_summary.py
import inspect

from anvil_code_agent.harness.loop import run, step


def test_step_and_run_accept_summarizer():
    assert "summarizer" in inspect.signature(step).parameters
    assert "summarizer" in inspect.signature(run).parameters
    assert inspect.signature(step).parameters["summarizer"].default is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_loop_summary.py -q`
Expected: FAIL(无 summarizer 参数)

- [ ] **Step 3: 实现(改 loop.py)**

`step` 的 import 行(`from anvil_code_agent.harness.context import compact`)与签名:给 `step` 和 `run` 各加 keyword-only 参 `summarizer: "Callable[..., str] | None" = None`,并把 `compact(...)` 调用补上 `summarizer=summarizer`。

`step` 内 compact 调用改为:

```python
        msgs = compact(msgs, max_tokens=token_budget, summarizer=summarizer)
```

`step` 签名追加 `summarizer=None`(keyword-only,放在 `token_budget` 后);`run` 签名同样追加 `summarizer=None`,并在调用 `step(...)` 时透传 `summarizer=summarizer`。导入用宽松注解避免新增 import:`summarizer=None` 不需要类型注解强约束,直接 `summarizer=None`。

具体:`step` 改为

```python
async def step(
    state: AgentState,
    model: str,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    policy: ApprovalPolicy = auto_approve,
    token_budget: int | None = None,
    summarizer=None,
) -> AgentState:
```
内部:
```python
    msgs = list(state.messages)
    if token_budget is not None:
        msgs = compact(msgs, max_tokens=token_budget, summarizer=summarizer)
```
`run` 改为
```python
async def run(
    state: AgentState,
    model: str,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    policy: ApprovalPolicy = auto_approve,
    token_budget: int | None = None,
    summarizer=None,
) -> AgentState:
```
内部 step 调用:
```python
            state = await step(
                state, model, registry, ctx,
                policy=policy, token_budget=token_budget, summarizer=summarizer,
            )
```

- [ ] **Step 4: 跑测试确认通过 + 回归 loop 测试**

Run: `uv run pytest packages/code-agent/tests/test_loop_summary.py packages/code-agent/tests/test_loop_m3.py packages/code-agent/tests/test_loop_step.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/harness/loop.py packages/code-agent/tests/test_loop_summary.py
git commit -m "feat(code-agent): thread summarizer through step/run (default None = M3 behavior)"
```

---

## Task 3: repo map 符号级排名(按被引次数,top-K)

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/repomap/build.py`
- Test: `packages/code-agent/tests/test_repomap_symbols.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_repomap_symbols.py
from anvil_code_agent.repomap.build import build_repo_map


def test_symbols_ranked_by_reference_count(tmp_path):
    # core.py 定义 hot(被多处调用)和 cold(没人调用);hot 应排在 cold 前
    (tmp_path / "core.py").write_text("def hot(x):\n    return x\n\ndef cold(y):\n    return y\n")
    (tmp_path / "a.py").write_text("from core import hot\n\ndef fa():\n    return hot(1)\n")
    (tmp_path / "b.py").write_text("from core import hot\n\ndef fb():\n    return hot(hot(2))\n")
    text = build_repo_map(str(tmp_path), ["core.py", "a.py", "b.py"], max_chars=4000)
    # 在 core.py 段里 hot 出现在 cold 之前
    seg = text[text.index("core.py"):]
    assert seg.index("hot") < seg.index("cold")


def test_per_file_symbol_cap(tmp_path):
    # 一个文件很多 def,渲染应限制每文件符号数(top-K)并标省略
    body = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(40))
    (tmp_path / "big.py").write_text(body)
    text = build_repo_map(str(tmp_path), ["big.py"], max_chars=4000, max_symbols_per_file=8)
    # big.py 段里最多 8 个符号 + 省略提示
    assert "more symbols" in text or "…" in text or "..." in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_repomap_symbols.py -q`
Expected: FAIL(无符号排名 / max_symbols_per_file 参数)

- [ ] **Step 3: 实现(改 build.py)**

给 `build_repo_map` 加 `max_symbols_per_file: int = 30` 参;计算 `ref_count`(每个 name 全仓被引次数);渲染每文件时按 ref_count 降序取 top-K。把第 1 步抽 tags、第 5 步渲染之间补 ref_count,渲染块改为符号排名:

```python
def build_repo_map(
    root: str, files: list[str], *, max_chars: int = 4000, max_symbols_per_file: int = 30
) -> str:
    if not files:
        return ""
    file_defs: dict[str, set[str]] = {}
    file_refs: dict[str, list[str]] = {}
    for rel in files:
        full = os.path.join(root, rel)
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                code = fh.read()
        except OSError:
            continue
        t = extract_tags(code)
        file_defs[rel] = t.defs
        file_refs[rel] = t.refs
    definers: dict[str, set[str]] = {}
    for rel, defs in file_defs.items():
        for d in defs:
            definers.setdefault(d, set()).add(rel)
    graph: dict[str, dict[str, float]] = {rel: {} for rel in file_defs}
    ref_count: dict[str, int] = {}
    for rel, refs in file_refs.items():
        for r in refs:
            ref_count[r] = ref_count.get(r, 0) + 1
            for dst in definers.get(r, ()):
                if dst != rel:
                    graph[rel][dst] = graph[rel].get(dst, 0.0) + 1.0
    ranks = pagerank(graph)
    ranked = sorted(file_defs, key=lambda f: ranks.get(f, 0.0), reverse=True)
    lines: list[str] = []
    out_len = 0
    for rel in ranked:
        # symbols ranked by how often they're referenced across the repo (hubs first)
        syms = sorted(file_defs.get(rel, set()), key=lambda d: (-ref_count.get(d, 0), d))
        shown = syms[:max_symbols_per_file]
        block = f"{rel}:\n" + "".join(f"  {d}\n" for d in shown)
        if len(syms) > max_symbols_per_file:
            block += f"  ... ({len(syms) - max_symbols_per_file} more symbols)\n"
        if out_len + len(block) > max_chars:
            if not lines:
                header = f"{rel}:\n"
                budget_left = max_chars - len(header)
                fit: list[str] = []
                for d in shown:
                    line = f"  {d}\n"
                    if budget_left - len(line) < 0:
                        break
                    fit.append(line)
                    budget_left -= len(line)
                lines.append(header + "".join(fit))
            lines.append(f"... [repo map truncated at {max_chars} chars]")
            break
        lines.append(block)
        out_len += len(block)
    return "".join(lines)
```

- [ ] **Step 4: 跑测试确认通过 + 回归 repomap 测试**

Run: `uv run pytest packages/code-agent/tests/test_repomap_symbols.py packages/code-agent/tests/test_repomap_build.py packages/code-agent/tests/test_tools_repomap.py -q`
Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/repomap/build.py packages/code-agent/tests/test_repomap_symbols.py
git commit -m "feat(code-agent): symbol-level repo map ranking (by reference count, top-K)"
```

---

## Task 4: 文档 + 全量回归

**Files:**
- Modify: `anvil/CLAUDE.md`、`examples/06-code-agent/README.md`

- [ ] **Step 1: 全量回归(根目录,含全仓 ruff)**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest -m "not live" -q && uv run ruff check .`
Expected: 全绿,ruff 净(注意:CI 跑 `ruff check .` 全仓——本地也用 `.` 而非只 packages/code-agent)

- [ ] **Step 2: 写文档**

`anvil/CLAUDE.md` 的 anvil-code-agent 段补:

```markdown
- `context.compact` 摘要 tier(M6)— 截断后仍超预算时,把"中间回合"整段替换成一条 LLM 摘要(`llm_summarizer(model)`,gateway 实现),替换边界对齐非 tool 消息保 tool_use 配对;step/run 加 `summarizer` 可选参(默认 None=只截断)
- `repo_map` 符号级排名(M6)— 每文件符号按全仓被引次数降序(枢纽符号优先)+ 每文件 top-K(`max_symbols_per_file`),比 M2 的文件级更细
```

`examples/06-code-agent/README.md` 末尾追加:

```markdown
## CA-M6:更深的上下文工程 + 检索

- **摘要压缩 tier**:M3 的 compact 只会截断老工具输出;M6 加一层——超预算时把"中间回合"(系统/任务/最近窗口之外)整段换成一条 LLM 摘要,长任务下 context 不爆且保留要点。关键难点在**不破坏 tool_use 配对**:替换边界对齐到非 tool 消息,孤儿 tool 消息不会出现。
- **符号级 repo map**:M2 排到文件粒度;M6 进一步把每个文件里的符号按"全仓被引次数"排序(被调最多的枢纽函数/类优先),并每文件限 top-K——agent 一眼看到最该关注的符号。
```

- [ ] **Step 3: 提交**

```bash
git add CLAUDE.md examples/06-code-agent/README.md
git commit -m "docs(code-agent): CA-M6 context summary tier + symbol-level repo map"
```

---

## 完成标准(CA-M6 验收)

- `uv run pytest packages/code-agent -m "not live" -q` 全绿;`uv run ruff check .`(全仓)净;根目录全量无 collision。
- compact 摘要 tier:超预算 + 有 summarizer 时替换中间回合为摘要,**tool_use 配对完好**;默认 None 时同 M3 只截断。
- repo map 符号按被引次数排(hot 在 cold 前)+ 每文件 top-K。
- step/run 的 summarizer 默认 None,M3 行为不变。
- 触底:上下文工程(摘要压缩)+ Aider 式符号级 repo map。
