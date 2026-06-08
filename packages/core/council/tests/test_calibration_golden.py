from pathlib import Path

from anvil_eval.calibration import load_calibration

CALIB = (
    Path(__file__).resolve().parents[3]
    / "core"
    / "eval"
    / "golden"
    / "calibration.jsonl"
)


def test_calibration_set_is_large_enough():
    cases = load_calibration(str(CALIB))
    assert len(cases) >= 30, f"expected >=30 calibration cases, got {len(cases)}"


def test_calibration_scores_span_three_buckets():
    cases = load_calibration(str(CALIB))
    scores = {c.human_score for c in cases}
    assert 1.0 in scores and 0.5 in scores and 0.0 in scores


def test_calibration_ids_unique():
    cases = load_calibration(str(CALIB))
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
