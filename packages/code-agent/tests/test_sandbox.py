import os
import subprocess

from anvil_code_agent.sandbox import Worktree


def _init_repo(p):
    subprocess.run(["git", "init", "-q"], cwd=p, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=p, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=p, check=True)
    (p / "f.txt").write_text("orig\n")
    subprocess.run(["git", "add", "."], cwd=p, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=p, check=True)


def test_worktree_creates_isolated_copy(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    with Worktree(str(repo)) as wt:
        assert os.path.isfile(os.path.join(wt.path, "f.txt"))
        # 改动隔离:在 worktree 改不影响原仓工作区
        with open(os.path.join(wt.path, "f.txt"), "w") as fh:
            fh.write("changed\n")
        assert (repo / "f.txt").read_text() == "orig\n"


def test_worktree_diff_captures_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    with Worktree(str(repo)) as wt:
        with open(os.path.join(wt.path, "f.txt"), "w") as fh:
            fh.write("changed\n")
        d = wt.diff()
        assert "changed" in d and "orig" in d


def test_worktree_cleanup_removes_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    with Worktree(str(repo)) as wt:
        path = wt.path
    assert not os.path.exists(path)
