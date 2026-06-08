from pathlib import Path

import pytest
from anvil_eval.calibration import (
    CalibrationReport,
    cohen_kappa,
    load_calibration,
    quantize,
)

CALIB = Path(__file__).resolve().parents[1] / "golden" / "calibration.jsonl"


def test_quantize_three_buckets():
    assert quantize(0.0) == 0
    assert quantize(0.2) == 0
    assert quantize(0.5) == 1
    assert quantize(0.6) == 1
    assert quantize(0.7) == 2
    assert quantize(1.0) == 2


def test_cohen_kappa_hand_computed():
    # a=[2,2,0,1], b=[2,0,0,1]: po=3/4=0.75; categories {0,1,2}
    # pa=(0.25,0.25,0.5) pb=(0.5,0.25,0.25); pe=0.25*0.5+0.25*0.25+0.5*0.25=0.3125
    # kappa=(0.75-0.3125)/(1-0.3125)=0.4375/0.6875=0.63636...
    k = cohen_kappa([2, 2, 0, 1], [2, 0, 0, 1])
    assert abs(k - 0.63636) < 0.001


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa([0, 1, 2], [0, 1, 2]) == 1.0


def test_cohen_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa([0, 1], [0])


def test_load_calibration_has_human_scores():
    cases = load_calibration(str(CALIB))
    assert len(cases) >= 12
    for c in cases:
        assert 0.0 <= c.human_score <= 1.0
        assert c.answer  # candidate answer present
        assert c.question


def test_calibration_report_shape():
    report = CalibrationReport(kappa=0.55, n=14, judge_labels=[1] * 14, human_labels=[1] * 14)
    md = report.to_markdown()
    assert "kappa" in md.lower() or "κ" in md
    assert "14" in md
