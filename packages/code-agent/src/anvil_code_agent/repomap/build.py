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
