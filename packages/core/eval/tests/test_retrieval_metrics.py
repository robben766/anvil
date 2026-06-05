"""检索指标单元测试 — 手算锁定,无 LLM 调用。"""

import pytest
from anvil_eval.metrics.retrieval import mrr, precision_at_k, recall_at_k

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


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


def test_mrr_rank1_hit():
    """第1位命中 → MRR = 1.0。"""
    evidences = ["等待期为90天"]
    retrieved = ["本合同等待期为90天,自合同成立起计算。"]
    assert mrr(retrieved, evidences) == pytest.approx(1.0)


def test_mrr_rank3_hit():
    """第3位命中 → MRR = 1/3 ≈ 0.3333。"""
    evidences = ["等待期为90天"]
    retrieved = [
        "第一章:定义",
        "第二章:责任",
        "本合同等待期为90天,宽限期60天。",  # rank 3
    ]
    assert mrr(retrieved, evidences) == pytest.approx(1 / 3)


def test_mrr_all_miss():
    """全不命中 → MRR = 0.0。"""
    evidences = ["等待期为90天"]
    retrieved = ["完全无关内容A", "完全无关内容B", "完全无关内容C"]
    assert mrr(retrieved, evidences) == pytest.approx(0.0)


def test_mrr_empty_evidences_returns_none():
    """evidences=[] → mrr 返回 None。"""
    retrieved = ["chunk1", "chunk2"]
    assert mrr(retrieved, []) is None


def test_mrr_empty_retrieved_with_evidences():
    """retrieved_texts=[] 且有 evidences → mrr 返回 0.0。"""
    evidences = ["等待期为90天"]
    assert mrr([], evidences) == pytest.approx(0.0)


def test_mrr_first_hit_wins_over_later():
    """多 evidence,第2位命中其中1个 → 1/2;不用等第1个 evidence 命中才算。"""
    evidences = ["等待期为90天", "保额50万"]
    retrieved = [
        "完全无关内容",              # rank 1, miss
        "保额50万一次性给付。",       # rank 2, hits evidence[1] (子串完全包含)
        "等待期为90天起计算。",       # rank 3, hits evidence[0]
    ]
    # First hit is at rank 2 → 1/2
    assert mrr(retrieved, evidences) == pytest.approx(1 / 2)


def test_mrr_whitespace_normalization():
    """空白差异命中:chunk 含 '等待期 为 90天',evidence '等待期为90天' → rank1 → 1.0。"""
    evidences = ["等待期为90天"]
    retrieved = ["等待期 为 90天,自合同成立起计算。"]
    assert mrr(retrieved, evidences) == pytest.approx(1.0)
