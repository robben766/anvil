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


def test_all_zero_weights_returns_uniform():
    # When all edge weights are zero, out_weight is 0 for all nodes so every node
    # is dangling — the convergence path still works, but the total must be non-zero.
    # With all-zero weights the guard must prevent a division-by-zero.
    pr = pagerank({"A": {"B": 0.0}, "B": {}})
    assert pr is not None
    total = sum(pr.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    # Both values must be finite and positive
    assert pr["A"] > 0
    assert pr["B"] > 0
