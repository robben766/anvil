import json

import pytest
from anvil_eval.dataset import GoldenCase, load_dataset

VALID = {
    "id": "c1",
    "question": "等待期是多少天?",
    "reference": "等待期为 90 天。",
    "contexts": ["第 5 条:本合同等待期为 90 天。"],
    "tags": ["policy"],
}


def _write(tmp_path, rows):
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(p)


def test_load_valid(tmp_path):
    cases = load_dataset(_write(tmp_path, [VALID]))
    assert len(cases) == 1
    c = cases[0]
    assert isinstance(c, GoldenCase)
    assert c.id == "c1" and c.contexts == ["第 5 条:本合同等待期为 90 天。"]


def test_contexts_and_tags_optional(tmp_path):
    row = {"id": "c2", "question": "q", "reference": "r"}
    c = load_dataset(_write(tmp_path, [row]))[0]
    assert c.contexts == [] and c.tags == []


def test_missing_required_field_raises(tmp_path):
    with pytest.raises(ValueError, match="reference"):
        load_dataset(_write(tmp_path, [{"id": "c3", "question": "q"}]))


def test_duplicate_id_raises(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(_write(tmp_path, [VALID, VALID]))
