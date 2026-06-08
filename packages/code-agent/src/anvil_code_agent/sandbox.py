"""git worktree isolation: each task runs in a throwaway worktree off the target repo.
Changes are isolated, capturable via diff(), and discarded on cleanup."""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid


class Worktree:
    def __init__(self, repo: str):
        self.repo = os.path.abspath(repo)
        self.path = os.path.join(tempfile.gettempdir(), f"anvil-wt-{uuid.uuid4().hex[:8]}")
        self._branch = f"anvil/wt-{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> Worktree:
        subprocess.run(
            ["git", "worktree", "add", "-q", "-b", self._branch, self.path, "HEAD"],
            cwd=self.repo,
            check=True,
        )
        return self

    def diff(self) -> str:
        return subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def __exit__(self, *exc) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", self.path],
            cwd=self.repo,
            capture_output=True,
        )
        subprocess.run(["git", "branch", "-D", self._branch], cwd=self.repo, capture_output=True)
