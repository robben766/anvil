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


def test_invalid_json_reports_file_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"c1","question":"q","reference":"r"}\n{bad json}', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_dataset(str(p))


def test_load_evidences_and_answerable(tmp_path):
    """新字段 evidences / answerable 能加载且值正确。"""
    row = {
        "id": "c10",
        "question": "等待期是多少?",
        "reference": "90天",
        "evidences": ["等待期为90天", "自合同成立起90天"],
        "answerable": False,
    }
    c = load_dataset(_write(tmp_path, [row]))[0]
    assert c.evidences == ["等待期为90天", "自合同成立起90天"]
    assert c.answerable is False


def test_old_format_missing_new_fields(tmp_path):
    """旧格式行(无 evidences/answerable)加载后使用默认值。"""
    row = {"id": "c11", "question": "q", "reference": "r"}
    c = load_dataset(_write(tmp_path, [row]))[0]
    assert c.evidences == []
    assert c.answerable is True
