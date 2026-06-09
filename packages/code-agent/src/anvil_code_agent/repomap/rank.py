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
    if total == 0.0:
        return {v: 1.0 / n for v in nodes}
    return {v: rank[v] / total for v in nodes}
