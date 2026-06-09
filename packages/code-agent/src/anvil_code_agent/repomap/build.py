"""Build an Aider-style repo map: parse tags per file, build a 'referencing file →
defining file' graph, PageRank it, render the top files with their definitions
within a character budget. M6: symbols within each file are ranked by repo-wide
reference count (hub symbols first), capped at max_symbols_per_file."""

from __future__ import annotations

import os

from anvil_code_agent.repomap.rank import pagerank
from anvil_code_agent.repomap.tags import extract_tags


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
