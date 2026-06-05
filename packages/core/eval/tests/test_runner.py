import pytest
from anvil_eval.dataset import GoldenCase
from anvil_eval.runner import EvalReport, run_eval


@pytest.fixture
def stub_metrics(monkeypatch):
    async def f(**kw):
        return 0.9

    async def low(**kw):
        return 0.4

    monkeypatch.setattr("anvil_eval.runner.faithfulness", f)
    monkeypatch.setattr("anvil_eval.runner.context_recall", f)
    monkeypatch.setattr("anvil_eval.runner.answer_relevancy", low)
    monkeypatch.setattr("anvil_eval.runner.context_precision", f)


CASES = [GoldenCase(id="c1", question="q", reference="r", contexts=["x"])]


async def _answer(case):
    return "ans"


async def test_report_aggregates_means(stub_metrics):
    report = await run_eval(CASES, _answer)
    assert isinstance(report, EvalReport)
    assert report.means["faithfulness"] == pytest.approx(0.9)
    assert report.means["answer_relevancy"] == pytest.approx(0.4)
    assert report.overall == pytest.approx((0.9 + 0.9 + 0.4 + 0.9) / 4)


async def test_threshold_pass_fail(stub_metrics):
    report = await run_eval(CASES, _answer)
    assert report.passed(threshold=0.7)
    assert not report.passed(threshold=0.8)


async def test_markdown_report_contains_table(stub_metrics):
    report = await run_eval(CASES, _answer)
    md = report.to_markdown()
    assert "faithfulness" in md and "c1" in md and "overall" in md
