from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.search import grep


def test_grep_finds_matches_with_path_and_line(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ndef target():\n    pass\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    res = grep({"pattern": "target"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "a.py:2" in res.content
    assert "def target" in res.content


def test_grep_no_match_is_readable_not_error(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    res = grep({"pattern": "zzz"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok  # 无匹配不是错误
    assert "no matches" in res.content.lower()


def test_grep_skips_dot_git(tmp_path):
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    (gitdir / "config").write_text("target\n")
    (tmp_path / "a.py").write_text("target\n")
    res = grep({"pattern": "target"}, ToolContext(workdir=str(tmp_path)))
    assert "a.py" in res.content
    assert ".git" not in res.content


def test_grep_truncates(tmp_path):
    (tmp_path / "big.py").write_text("\n".join("target" for _ in range(5000)))
    res = grep({"pattern": "target"}, ToolContext(workdir=str(tmp_path), max_output=300))
    assert res.truncated
