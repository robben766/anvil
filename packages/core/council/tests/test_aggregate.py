import pytest
from anvil_council.aggregate import aggregate
from anvil_council.rubric import DEFAULT_RUBRIC
from anvil_council.verdict import JurorScore, Verdict


def _js(model, correctness, evidence, completeness, relevance, overall):
    return JurorScore(
        model=model,
        per_criterion={
            "correctness": correctness,
            "evidence": evidence,
            "completeness": completeness,
            "relevance": relevance,
        },
        overall=overall,
        reason="r",
    )


def test_aggregate_medians_and_disagreement():
    a = _js("deepseek-chat", 1.0, 1.0, 1.0, 1.0, 1.0)
    b = _js("qwen-plus", 0.0, 1.0, 1.0, 1.0, 0.5)
    v = aggregate([a, b], DEFAULT_RUBRIC, disagreement_threshold=0.5)
    assert isinstance(v, Verdict)
    assert v.per_criterion["correctness"] == pytest.approx(0.5)
    assert v.per_criterion["evidence"] == pytest.approx(1.0)
    assert v.overall == pytest.approx(0.75)
    assert v.disagreements == ["correctness"]
    assert v.confidence == pytest.approx(0.75)
    assert len(v.jurors) == 2


def test_aggregate_unanimous_no_disagreement_full_confidence():
    a = _js("deepseek-chat", 1.0, 1.0, 1.0, 1.0, 1.0)
    b = _js("qwen-plus", 1.0, 1.0, 1.0, 1.0, 1.0)
    v = aggregate([a, b], DEFAULT_RUBRIC)
    assert v.disagreements == []
    assert v.confidence == pytest.approx(1.0)
    assert v.overall == pytest.approx(1.0)


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate([], DEFAULT_RUBRIC)
