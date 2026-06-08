from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.fs import edit_file


def _ctx(p):
    return ToolContext(workdir=str(p))


def test_edit_applies_unique_match(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def add(a, b):\n    return a - b\n")
    res = edit_file(
        {"path": "a.py", "search": "    return a - b", "replace": "    return a + b"},
        _ctx(tmp_path),
    )
    assert res.ok
    assert f.read_text() == "def add(a, b):\n    return a + b\n"


def test_edit_not_found_is_error_and_no_write(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    res = edit_file({"path": "a.py", "search": "y = 2", "replace": "y = 3"}, _ctx(tmp_path))
    assert res.ok is False
    assert "not found" in res.content.lower()
    assert f.read_text() == "x = 1\n"  # 未改


def test_edit_multiple_matches_is_error(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("v = 1\nv = 1\n")
    res = edit_file({"path": "a.py", "search": "v = 1", "replace": "v = 2"}, _ctx(tmp_path))
    assert res.ok is False
    assert "multiple" in res.content.lower() or "2 matches" in res.content.lower()
    assert f.read_text() == "v = 1\nv = 1\n"


def test_edit_empty_search_is_error(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    res = edit_file({"path": "a.py", "search": "", "replace": "y = 2"}, _ctx(tmp_path))
    assert res.ok is False
    assert "empty" in res.content.lower()


def test_edit_missing_file_is_error(tmp_path):
    res = edit_file({"path": "no.py", "search": "a", "replace": "b"}, _ctx(tmp_path))
    assert res.ok is False
    assert "not found" in res.content.lower()
