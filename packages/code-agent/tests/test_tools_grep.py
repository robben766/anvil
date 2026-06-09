import os

from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.search import _MAX_FILE_BYTES, grep


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


def test_grep_skips_binary_file(tmp_path):
    # A file containing a null byte is treated as binary and skipped entirely.
    # The pattern text appears AFTER the null byte, so it would match if parsed
    # as text — but must not appear in results.
    (tmp_path / "binary.bin").write_bytes(b"hello\x00target_token\n")
    res = grep({"pattern": "target_token"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "binary.bin" not in res.content


def test_grep_skips_large_file(tmp_path):
    # A file larger than _MAX_FILE_BYTES must be skipped.
    large_file = tmp_path / "large.py"
    # Write a file just over the cap with a line containing the search term
    chunk = b"target_in_large_file\n" * 100
    with open(large_file, "wb") as f:
        written = 0
        while written <= _MAX_FILE_BYTES:
            f.write(chunk)
            written += len(chunk)
    assert os.path.getsize(large_file) > _MAX_FILE_BYTES
    res = grep({"pattern": "target_in_large_file"}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "large.py" not in res.content
