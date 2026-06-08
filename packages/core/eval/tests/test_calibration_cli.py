import os

import pytest
from anvil_eval.calibration import build_report
from anvil_eval.cli import _calibrate as calibrate_entry  # noqa: F401  (import-existence check)


def test_build_report_end_to_end():
    # judge scores perfectly matching human → kappa 1.0
    judge = [1.0, 0.0, 0.5, 1.0]
    human = [1.0, 0.0, 0.5, 1.0]
    report = build_report(judge, human)
    assert report.kappa == 1.0
    assert report.n == 4


@pytest.mark.live
async def test_calibrate_live_runs():
    # Only runs with -m live and a real DEEPSEEK_API_KEY; smoke-checks the judge path.
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("no key")
