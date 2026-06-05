import numpy as np
import pytest
from anvil_eval.metrics.answer_relevancy import _mean_cosine
from anvil_eval.metrics.context_precision import precision_from_flags


def test_precision_hand_calculated_ragas_example():
    # RAGAS 文档算例:[相关, 不相关, 相关, 不相关]
    # P@1=1/1(v=1), P@3=2/3(v=1) → (1 + 2/3) / 2 = 0.8333…
    assert precision_from_flags([True, False, True, False]) == pytest.approx(0.8333, abs=1e-3)


def test_precision_all_relevant_is_one():
    assert precision_from_flags([True, True, True]) == pytest.approx(1.0)


def test_precision_none_relevant_is_zero():
    assert precision_from_flags([False, False]) == 0.0


def test_mean_cosine_identical_vectors_is_one():
    v = np.array([[0.6, 0.8]])
    assert _mean_cosine(np.array([0.6, 0.8]), v) == pytest.approx(1.0)


def test_mean_cosine_orthogonal_is_zero():
    q = np.array([1.0, 0.0])
    others = np.array([[0.0, 1.0]])
    assert _mean_cosine(q, others) == pytest.approx(0.0, abs=1e-9)
