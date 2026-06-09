from anvil_code_agent.tools.base import ToolContext
from anvil_code_agent.tools.search import repo_map


def test_repo_map_lists_python_symbols(tmp_path):
    (tmp_path / "core.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "a.py").write_text("from core import helper\n\ndef fa():\n    return helper(1)\n")
    res = repo_map({}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "core.py" in res.content
    assert "helper" in res.content


def test_repo_map_empty_repo(tmp_path):
    res = repo_map({}, ToolContext(workdir=str(tmp_path)))
    assert res.ok
    assert "no python files" in res.content.lower() or res.content == ""


def test_repo_map_skips_vcs_dirs(tmp_path):
    g = tmp_path / ".git"
    g.mkdir()
    (g / "hook.py").write_text("def sneaky():\n    pass\n")
    (tmp_path / "real.py").write_text("def real_fn():\n    pass\n")
    res = repo_map({}, ToolContext(workdir=str(tmp_path)))
    assert "real.py" in res.content
    assert "sneaky" not in res.content
