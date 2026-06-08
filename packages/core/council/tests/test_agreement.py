import pytest
from anvil_council.agreement import fleiss_kappa, jurors_fleiss


def test_fleiss_kappa_hand_computed():
    # 2 items, 3 raters, 2 categories.
    # item1 = [3,0] (all raters cat0); item2 = [1,2].
    # p = (0.667, 0.333); P_e = 0.5556
    # P_i = [1.0, 0.333]; P_bar = 0.6667
    # kappa = (0.6667-0.5556)/(1-0.5556) = 0.25
    k = fleiss_kappa([[3, 0], [1, 2]])
    assert abs(k - 0.25) < 0.01


def test_fleiss_perfect_agreement():
    assert fleiss_kappa([[3, 0], [0, 3]]) == 1.0


def test_fleiss_rejects_ragged_rows():
    with pytest.raises(ValueError):
        fleiss_kappa([[3, 0], [1, 1, 1]])


def test_jurors_fleiss_builds_matrix():
    # item1 jurors all 0 → [3,0]; item2 jurors [0,1,1] → [1,2]  → same as hand-computed 0.25
    k = jurors_fleiss([[0, 0, 0], [0, 1, 1]], num_categories=2)
    assert abs(k - 0.25) < 0.01
