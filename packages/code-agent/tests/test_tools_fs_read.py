from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.fs import read_file


def test_read_returns_numbered_lines(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n")
    res = read_file({"path": "a.py"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "1\tx = 1" in res.content
    assert "2\ty = 2" in res.content


def test_read_missing_file_is_error(tmp_path):
    res = read_file({"path": "nope.py"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok is False
    assert "not found" in res.content.lower()


def test_read_truncates_long_file(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line{i}" for i in range(10000)))
    res = read_file({"path": "big.txt"}, ToolContext(workdir=str(tmp_path), max_output=200))
    assert res.truncated
    assert len(res.content) <= 400  # 截断 + 提示行的宽松上界
