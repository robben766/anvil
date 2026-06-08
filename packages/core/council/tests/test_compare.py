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
    md = rep.to_markdown()
    assert "jury" in md.lower()


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
