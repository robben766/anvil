"""检索指标单元测试 — 手算锁定,无 LLM 调用。"""

import pytest
from anvil_eval.metrics.retrieval import precision_at_k, recall_at_k

# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_basic():
    """3 evidences,top-3 覆盖其中 2 → 2/3。"""
    evidences = ["等待期为90天", "免赔额为1000元", "保额50万"]
    retrieved = [
        "本合同等待期为90天,自合同成立起计算。",   # 覆盖 evidence[0]
        "本合同免赔额为1000元整。",               # 覆盖 evidence[1]
        "本产品不含意外险。",                     # 不含 evidence[2]
    ]
    result = recall_at_k(retrieved, evidences, k=3)
    assert result == pytest.approx(2 / 3)


def test_precision_at_k_basic():
    """3 个 chunk 中 2 个命中任一 evidence → 2/3。"""
    evidences = ["等待期为90天", "免赔额为1000元"]
    retrieved = [
        "本合同等待期为90天,自合同成立起计算。",  # 命中 evidence[0]
        "本合同免赔额为1000元整。",               # 命中 evidence[1]
        "本产品不含意外险。",                     # 不命中
    ]
    result = precision_at_k(retrieved, evidences, k=3)
    assert result == pytest.approx(2 / 3)


def test_recall_k_truncation():
    """k 截断:第 4 位才命中的 evidence 在 k=3 不计。"""
    evidences = ["保额50万"]
    retrieved = [
        "第一章:定义",
        "第二章:责任",
        "第三章:除外",
        "保额为50万元。",  # 第 4 位,k=3 不看
    ]
    result = recall_at_k(retrieved, evidences, k=3)
    assert result == pytest.approx(0.0)


def test_recall_at_k_empty_evidences():
    """evidences=[] → recall 返回 None。"""
    retrieved = ["chunk1", "chunk2"]
    assert recall_at_k(retrieved, [], k=3) is None


def test_precision_at_k_empty_evidences():
    """evidences=[] → precision 返回 None。"""
    retrieved = ["chunk1", "chunk2"]
    assert precision_at_k(retrieved, [], k=3) is None


def test_whitespace_normalization_hit():
    """空白差异命中:chunk 含 '等待期 为 90天',evidence '等待期为90天' → 归一化后命中。"""
    evidences = ["等待期为90天"]
    retrieved = ["等待期 为 90天,自合同成立起计算。"]
    assert recall_at_k(retrieved, evidences, k=1) == pytest.approx(1.0)
    assert precision_at_k(retrieved, evidences, k=1) == pytest.approx(1.0)


def test_recall_empty_retrieved_with_evidences():
    """retrieved_texts=[] 且有 evidences → recall 0.0。"""
    evidences = ["等待期为90天"]
    assert recall_at_k([], evidences, k=3) == pytest.approx(0.0)


def test_precision_empty_retrieved_with_evidences():
    """retrieved_texts=[] 且有 evidences → precision 0.0。"""
    evidences = ["等待期为90天"]
    assert precision_at_k([], evidences, k=3) == pytest.approx(0.0)
