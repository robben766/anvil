import math

import pytest
from anvil_council.agreement import CompareReport, compare_jury


def test_compare_jury_perfect_alignment():
    juror_overalls = {
        "deepseek-chat": [1.0, 0.0, 0.5, 1.0],
        "qwen-plus": [1.0, 0.0, 0.5, 1.0],
    }
    jury_overalls = [1.0, 0.0, 0.5, 1.0]
    human = [1.0, 0.0, 0.5, 1.0]
    rep = compare_jury(juror_overalls, jury_overalls, human)
    assert isinstance(rep, CompareReport)
    assert rep.jury_vs_human == 1.0
    assert rep.single_vs_human["deepseek-chat"] == 1.0
    assert rep.best_single_kappa == 1.0
    assert rep.n == 4
    # both jurors gave identical scores → perfect inter-juror agreement
    assert rep.inter_juror_fleiss == pytest.approx(1.0)
    md = rep.to_markdown()
    assert "jury" in md.lower()
    assert "fleiss" in md.lower()


def test_compare_jury_single_juror_fleiss_is_nan():
    # one juror → inter-juror agreement is undefined (nan), not a crash
    rep = compare_jury({"solo": [1.0, 0.0]}, [1.0, 0.0], [1.0, 0.0])
    assert math.isnan(rep.inter_juror_fleiss)
    assert "n/a" in rep.to_markdown()


def test_compare_jury_flags_when_jury_not_better():
    juror_overalls = {
        "good": [1.0, 0.0, 1.0, 0.0],
        "bad": [0.0, 1.0, 0.0, 1.0],
    }
    jury_overalls = [0.5, 0.5, 0.5, 0.5]
    human = [1.0, 0.0, 1.0, 0.0]
    rep = compare_jury(juror_overalls, jury_overalls, human)
    assert rep.best_single_kappa >= rep.jury_vs_human
    assert rep.jury_beats_best_single is False
