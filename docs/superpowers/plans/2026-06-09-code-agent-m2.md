# CA-M2 检索增强(repo map + grep)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 给编码 agent 装上"自己定位代码"的能力——tree-sitter 抽符号 + 手写 PageRank 排重要文件,渲染成精简 repo map;外加纯 Python agentic grep。让 agent 在多文件仓里不靠人喂文件。

**Architecture:** `repomap/tags.py` 用 tree-sitter 抽每个文件的定义(函数/类名)与引用(调用名);`repomap/rank.py` 手写 PageRank(含悬挂节点处理,手算对照);`repomap/build.py` 建"引用→定义"文件图、跑 PageRank、按预算渲染 top 文件+其定义;`tools/search.py` 把 `repo_map` 与 `grep` 包成工具接进默认注册表。

**Tech Stack:** tree-sitter 0.25 + tree-sitter-python 0.25(API:`Query(lang, s)` + `QueryCursor(q).captures(node) -> dict[name->[node]]`);手写 PageRank(不引 networkx);纯 Python grep(不依赖 ripgrep,CI 安全)。

**前置(已完成,随 Task 1 提交):** `packages/code-agent/pyproject.toml` 的 `[project].dependencies` 已加 `tree-sitter>=0.25` 与 `tree-sitter-python>=0.25`,`uv.lock` 已更新。

---

## 文件结构

- Create: `packages/code-agent/src/anvil_code_agent/repomap/__init__.py`(空)
- Create: `packages/code-agent/src/anvil_code_agent/repomap/tags.py` — tree-sitter 抽 defs/refs
- Create: `packages/code-agent/src/anvil_code_agent/repomap/rank.py` — 手写 PageRank
- Create: `packages/code-agent/src/anvil_code_agent/repomap/build.py` — 建图+排名+渲染
- Create: `packages/code-agent/src/anvil_code_agent/tools/search.py` — `grep` + `repo_map` 工具
- Modify: `packages/code-agent/src/anvil_code_agent/eval/runner.py` — 默认注册表加 repo_map+grep
- Modify: `packages/code-agent/pyproject.toml`(deps 已加)+ `CLAUDE.md` + `examples/06-code-agent/README.md`

---

## Task 1: tree-sitter 抽取 defs / refs

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/repomap/__init__.py`(空)
- Create: `packages/code-agent/src/anvil_code_agent/repomap/tags.py`
- Test: `packages/code-agent/tests/test_repomap_tags.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_repomap_tags.py
from anvil_code_agent.repomap.tags import extract_tags


def test_extract_defs_and_refs():
    code = (
        "def add(a, b):\n"
        "    return helper(a) + b\n"
        "\n"
        "class Calc:\n"
        "    def run(self):\n"
        "        return add(1, 2)\n"
        "\n"
        "def helper(x):\n"
        "    return x\n"
    )
    tags = extract_tags(code)
    assert tags.defs == {"add", "Calc", "run", "helper"}
    # 引用:helper、add 被调用(self/return 等不是 identifier-call)
    assert "helper" in tags.refs
    assert "add" in tags.refs


def test_empty_and_syntax_tolerant():
    tags = extract_tags("")
    assert tags.defs == set()
    assert tags.refs == []
    # 语法不完整也不崩(tree-sitter 容错)
    tags2 = extract_tags("def broken(:\n    x =")
    assert isinstance(tags2.defs, set)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/itachi/workspace/ai/anvil && uv run pytest packages/code-agent/tests/test_repomap_tags.py -q`
Expected: FAIL(ImportError: extract_tags)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/repomap/__init__.py
```

```python
# packages/code-agent/src/anvil_code_agent/repomap/tags.py
"""Extract definition and reference symbols from Python source via tree-sitter.
API note (tree-sitter 0.25): Query(lang, src) + QueryCursor(q).captures(node) -> dict."""

from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Query, QueryCursor

_PY = Language(tspy.language())
_PARSER = Parser(_PY)
_DEFS = Query(
    _PY,
    "(function_definition name: (identifier) @d) (class_definition name: (identifier) @d)",
)
_REFS = Query(_PY, "(call function: (identifier) @r)")


@dataclass
class Tags:
    defs: set[str]
    refs: list[str]


def _names(query: Query, root, source: bytes) -> list[str]:
    caps = QueryCursor(query).captures(root)
    out: list[str] = []
    for nodes in caps.values():
        out.extend(source[n.start_byte : n.end_byte].decode("utf-8", "replace") for n in nodes)
    return out


def extract_tags(code: str) -> Tags:
    source = code.encode("utf-8")
    root = _PARSER.parse(source).root_node
    defs = set(_names(_DEFS, root, source))
    refs = _names(_REFS, root, source)
    return Tags(defs=defs, refs=refs)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_repomap_tags.py -q`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/repomap packages/code-agent/tests/test_repomap_tags.py packages/code-agent/pyproject.toml
git add -f uv.lock 2>/dev/null; git add uv.lock
git commit -m "feat(code-agent): tree-sitter tag extraction (defs/refs) + deps"
```

---

## Task 2: 手写 PageRank(含悬挂节点,手算对照)

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/repomap/rank.py`
- Test: `packages/code-agent/tests/test_repomap_rank.py`

- [ ] **Step 1: 写失败测试(含手算对照)**

```python
# packages/code-agent/tests/test_repomap_rank.py
import pytest
from anvil_code_agent.repomap.rank import pagerank


def test_two_cycle_is_symmetric():
    # A->B->A 双向环:对称性 → 稳态严格 0.5/0.5(与阻尼无关),手算对照
    pr = pagerank({"A": {"B": 1.0}, "B": {"A": 1.0}})
    assert pr["A"] == pytest.approx(0.5, abs=1e-6)
    assert pr["B"] == pytest.approx(0.5, abs=1e-6)
    assert sum(pr.values()) == pytest.approx(1.0, abs=1e-9)


def test_referenced_node_ranks_highest():
    # A->C, B->C:C 被两处引用 → C 最高
    pr = pagerank({"A": {"C": 1.0}, "B": {"C": 1.0}, "C": {}})
    assert pr["C"] > pr["A"]
    assert pr["C"] > pr["B"]
    assert sum(pr.values()) == pytest.approx(1.0, abs=1e-9)


def test_dangling_node_redistributes():
    # 单悬挂节点(无出边)不该吞掉全部质量;两节点都在,和为 1
    pr = pagerank({"A": {"B": 1.0}, "B": {}})
    assert sum(pr.values()) == pytest.approx(1.0, abs=1e-9)
    assert pr["B"] > pr["A"]  # B 被指向且悬挂回流


def test_empty_graph():
    assert pagerank({}) == {}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_repomap_rank.py -q`
Expected: FAIL(ImportError: pagerank)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/repomap/rank.py
"""Hand-rolled PageRank over a weighted directed graph (no networkx).
graph[src][dst] = edge weight. Dangling nodes (no out-edges) redistribute their rank
uniformly each iteration. Returns a normalized distribution summing to 1.0."""

from __future__ import annotations


def pagerank(
    graph: dict[str, dict[str, float]],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> dict[str, float]:
    # node set = all sources + all destinations
    nodes: set[str] = set(graph)
    for dsts in graph.values():
        nodes.update(dsts)
    n = len(nodes)
    if n == 0:
        return {}
    rank = {v: 1.0 / n for v in nodes}
    out_weight = {v: sum(graph.get(v, {}).values()) for v in nodes}
    for _ in range(max_iter):
        dangling = sum(rank[v] for v in nodes if out_weight[v] == 0.0)
        new = {v: (1.0 - damping) / n + damping * dangling / n for v in nodes}
        for src in graph:
            ws = out_weight[src]
            if ws == 0.0:
                continue
            r = damping * rank[src]
            for dst, w in graph[src].items():
                new[dst] += r * (w / ws)
        delta = sum(abs(new[v] - rank[v]) for v in nodes)
        rank = new
        if delta < tol:
            break
    total = sum(rank.values())
    return {v: rank[v] / total for v in nodes}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_repomap_rank.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/repomap/rank.py packages/code-agent/tests/test_repomap_rank.py
git commit -m "feat(code-agent): hand-rolled PageRank with dangling redistribution"
```

---

## Task 3: build_repo_map(建图 + 排名 + 渲染)

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/repomap/build.py`
- Test: `packages/code-agent/tests/test_repomap_build.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_repomap_build.py
from anvil_code_agent.repomap.build import build_repo_map


def test_build_ranks_defining_file_and_lists_symbols(tmp_path):
    # core.py 定义 helper;a.py、b.py 都调用 helper → core.py 应排在最前
    (tmp_path / "core.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "a.py").write_text("from core import helper\n\ndef fa():\n    return helper(1)\n")
    (tmp_path / "b.py").write_text("from core import helper\n\ndef fb():\n    return helper(2)\n")
    text = build_repo_map(str(tmp_path), ["core.py", "a.py", "b.py"], max_chars=2000)
    # core.py 在 a.py / b.py 之前出现
    assert text.index("core.py") < text.index("a.py")
    assert text.index("core.py") < text.index("b.py")
    # 列出了定义符号
    assert "helper" in text


def test_build_respects_char_budget(tmp_path):
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text(f"def func{i}():\n    return {i}\n")
    files = [f"f{i}.py" for i in range(50)]
    text = build_repo_map(str(tmp_path), files, max_chars=300)
    assert len(text) <= 400  # 预算 + 截断提示的宽松上界


def test_build_handles_empty_filelist(tmp_path):
    assert build_repo_map(str(tmp_path), [], max_chars=2000) == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_repomap_build.py -q`
Expected: FAIL(ImportError: build_repo_map)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/repomap/build.py
"""Build an Aider-style repo map: parse tags per file, build a 'referencing file →
defining file' graph, PageRank it, render the top files with their definitions
within a character budget."""

from __future__ import annotations

import os

from anvil_code_agent.repomap.rank import pagerank
from anvil_code_agent.repomap.tags import extract_tags


def build_repo_map(root: str, files: list[str], *, max_chars: int = 4000) -> str:
    if not files:
        return ""
    # 1. 抽每个文件的 tags
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
    # 2. defname -> 定义它的文件集合
    definers: dict[str, set[str]] = {}
    for rel, defs in file_defs.items():
        for d in defs:
            definers.setdefault(d, set()).add(rel)
    # 3. 建图:引用文件 -> 定义文件(按引用次数加权)
    graph: dict[str, dict[str, float]] = {rel: {} for rel in file_defs}
    for rel, refs in file_refs.items():
        for r in refs:
            for dst in definers.get(r, ()):
                if dst != rel:
                    graph[rel][dst] = graph[rel].get(dst, 0.0) + 1.0
    # 4. 排名
    ranks = pagerank(graph)
    ranked = sorted(file_defs, key=lambda f: ranks.get(f, 0.0), reverse=True)
    # 5. 渲染到预算
    lines: list[str] = []
    out_len = 0
    for rel in ranked:
        defs = sorted(file_defs.get(rel, set()))
        block = f"{rel}:\n" + "".join(f"  {d}\n" for d in defs)
        if out_len + len(block) > max_chars:
            lines.append(f"... [repo map truncated at {max_chars} chars]")
            break
        lines.append(block)
        out_len += len(block)
    return "".join(lines)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_repomap_build.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/repomap/build.py packages/code-agent/tests/test_repomap_build.py
git commit -m "feat(code-agent): build_repo_map (ref->def graph + PageRank + budgeted render)"
```

---

## Task 4: grep 工具(纯 Python,CI 无依赖)

**Files:**
- Create: `packages/code-agent/src/anvil_code_agent/tools/search.py`
- Test: `packages/code-agent/tests/test_tools_grep.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_tools_grep.py
from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.search import grep


def test_grep_finds_matches_with_path_and_line(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ndef target():\n    pass\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    res = grep({"pattern": "target"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "a.py:2" in res.content
    assert "def target" in res.content


def test_grep_no_match_is_readable_not_error(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    res = grep({"pattern": "zzz"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok  # 无匹配不是错误
    assert "no matches" in res.content.lower()


def test_grep_skips_dot_git(tmp_path):
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    (gitdir / "config").write_text("target\n")
    (tmp_path / "a.py").write_text("target\n")
    res = grep({"pattern": "target"}, ToolContext(workdir=str(tmp_path)))
    assert "a.py" in res.content
    assert ".git" not in res.content


def test_grep_truncates(tmp_path):
    (tmp_path / "big.py").write_text("\n".join("target" for _ in range(5000)))
    res = grep({"pattern": "target"}, ToolContext(workdir=str(tmp_path), max_output=300))
    assert res.truncated
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_tools_grep.py -q`
Expected: FAIL(ImportError: grep)

- [ ] **Step 3: 实现**

```python
# packages/code-agent/src/anvil_code_agent/tools/search.py
"""Search tools: grep (pure-Python regex walk, CI-safe — no ripgrep dependency) and
repo_map (Aider-style symbol map to help the agent locate code)."""

from __future__ import annotations

import os
import re

from anvil_code_agent.repomap.build import build_repo_map
from anvil_code_agent.tools.base import ToolContext, ToolResult, tool

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


@tool(
    name="grep",
    description="Search file contents by regex. Returns 'relpath:lineno: line' matches. "
    "Paths are not sandboxed in M1 (workdir-relative walk).",
    params={
        "pattern": {"type": "string", "description": "Python regex"},
        "glob": {"type": "string", "description": "optional filename suffix filter, e.g. '.py'"},
    },
    required=["pattern"],
)
def grep(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        rx = re.compile(args["pattern"])
    except re.error as e:
        return ToolResult(content=f"invalid regex: {e}", ok=False)
    suffix = args.get("glob") or ""
    hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ctx.workdir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if suffix and not fn.endswith(suffix):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ctx.workdir)
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(f"{rel}:{i}: {line.rstrip()}")
            except OSError:
                continue
    if not hits:
        return ToolResult(content="no matches", ok=True)
    text = "\n".join(hits)
    if len(text) > ctx.max_output:
        return ToolResult(
            content=text[: ctx.max_output] + f"\n... [truncated, {len(hits)} matches]",
            ok=True,
            truncated=True,
        )
    return ToolResult(content=text, ok=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_tools_grep.py -q`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools/search.py packages/code-agent/tests/test_tools_grep.py
git commit -m "feat(code-agent): pure-Python grep tool (CI-safe, skips vcs dirs)"
```

---

## Task 5: repo_map 工具

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/tools/search.py`(追加 `repo_map`)
- Test: `packages/code-agent/tests/test_tools_repomap.py`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_tools_repomap.py
from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.search import repo_map


def test_repo_map_lists_python_symbols(tmp_path):
    (tmp_path / "core.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "a.py").write_text("from core import helper\n\ndef fa():\n    return helper(1)\n")
    res = repo_map({}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "core.py" in res.content
    assert "helper" in res.content


def test_repo_map_empty_repo(tmp_path):
    res = repo_map({}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "no python files" in res.content.lower() or res.content == ""


def test_repo_map_skips_vcs_dirs(tmp_path):
    g = tmp_path / ".git"
    g.mkdir()
    (g / "hook.py").write_text("def sneaky():\n    pass\n")
    (tmp_path / "real.py").write_text("def real_fn():\n    pass\n")
    res = repo_map({}, ToolContext(workdir=str(tmp_path)))
    assert "real.py" in res.content
    assert "sneaky" not in res.content
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_tools_repomap.py -q`
Expected: FAIL(ImportError: repo_map)

- [ ] **Step 3: 实现(追加到 search.py)**

```python
# 追加到 packages/code-agent/src/anvil_code_agent/tools/search.py 末尾


def _list_py_files(workdir: str) -> list[str]:
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.relpath(os.path.join(dirpath, fn), workdir))
    return files


@tool(
    name="repo_map",
    description="Return a ranked map of the repo's Python files and their top-level "
    "definitions (functions/classes), most-referenced files first. Use it to locate code.",
    params={},
    required=[],
)
def repo_map(args: dict, ctx: ToolContext) -> ToolResult:
    files = _list_py_files(ctx.workdir)
    if not files:
        return ToolResult(content="no python files found", ok=True)
    text = build_repo_map(ctx.workdir, files, max_chars=ctx.max_output)
    return ToolResult(content=text or "no python files found", ok=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest packages/code-agent/tests/test_tools_repomap.py -q`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add packages/code-agent/src/anvil_code_agent/tools/search.py packages/code-agent/tests/test_tools_repomap.py
git commit -m "feat(code-agent): repo_map tool (ranked symbol map for code location)"
```

---

## Task 6: 接进默认注册表 + 文档

**Files:**
- Modify: `packages/code-agent/src/anvil_code_agent/eval/runner.py`(default_registry 加 repo_map+grep)
- Test: `packages/code-agent/tests/test_registry_wired.py`
- Modify: `packages/code-agent/CLAUDE 段落`(见下)与 `examples/06-code-agent/README.md`

- [ ] **Step 1: 写失败测试**

```python
# packages/code-agent/tests/test_registry_wired.py
from anvil_code_agent.eval.runner import default_registry


def test_default_registry_includes_search_tools():
    names = [s["function"]["name"] for s in default_registry().schemas()]
    assert "read_file" in names
    assert "edit_file" in names
    assert "bash" in names
    assert "run_tests" in names
    assert "repo_map" in names
    assert "grep" in names
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest packages/code-agent/tests/test_registry_wired.py -q`
Expected: FAIL(repo_map/grep 不在)

- [ ] **Step 3: 实现(改 runner.py 的 import 与 default_registry)**

在 `runner.py` 顶部 import 区追加:

```python
from anvil_code_agent.tools.search import grep, repo_map
```

把 `default_registry` 改为:

```python
def default_registry() -> ToolRegistry:
    return ToolRegistry([read_file, edit_file, bash, run_tests, repo_map, grep])
```

- [ ] **Step 4: 跑测试确认通过 + 全包回归**

Run: `uv run pytest packages/code-agent -m "not live" -q && uv run ruff check packages/code-agent`
Expected: 全绿(约 65+ passed),ruff 净

- [ ] **Step 5: 文档 + 提交**

在 `anvil/CLAUDE.md` 的 anvil-code-agent 段落,工具行后补一行:

```markdown
- `repo_map` / `grep`(M2)— tree-sitter+手写 PageRank 排序的符号地图 + 纯 Python regex 搜索,让 agent 自己定位代码
```

在 `examples/06-code-agent/README.md` 末尾追加:

```markdown
## CA-M2:检索增强

agent 现在能自己定位代码,不靠人喂文件:
- `repo_map` — tree-sitter 抽每个 .py 的函数/类定义,手写 PageRank 把"被引用最多"的文件排前,渲染成预算内的符号地图(抄 Aider 的确定性方案)
- `grep` — 纯 Python regex 全仓搜索(跳过 .git/__pycache__ 等),CI 无外部依赖

设计要点:repo map 把"引用→定义"建成文件图跑 PageRank——定义了被广泛调用符号的文件(枢纽)自然浮到最前,正是 agent 最该先读的地方。
```

```bash
git add packages/code-agent/src/anvil_code_agent/eval/runner.py packages/code-agent/tests/test_registry_wired.py CLAUDE.md examples/06-code-agent/README.md
git commit -m "feat(code-agent): wire repo_map+grep into default registry + docs (CA-M2 complete)"
```

---

## 完成标准(CA-M2 验收)

- `uv run pytest packages/code-agent -m "not live" -q` 全绿;`uv run ruff check packages/code-agent` 净。
- 根目录 `uv run pytest -m "not live" -q` 全绿(无新 collision)。
- `repo_map` 把定义枢纽符号的文件排在最前;`grep` 跳过 vcs 目录、无匹配不报错。
- 默认注册表含 6 工具,agent 可在多文件仓自主定位代码。
- 触底:tree-sitter AST 抽取 + 手写 PageRank + Aider 式 repo map 渲染。
