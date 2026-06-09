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


def has_docker() -> bool:
    """True if a docker daemon is reachable."""
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerSandbox:
    """Process-isolated sandbox: a container with the work dir bind-mounted at /work.
    Edit files on the host path; run commands inside the container via exec()."""

    def __init__(self, workdir: str, image: str = "python:3.12-slim"):
        self.workdir = os.path.abspath(workdir)
        self.image = image
        self.name = f"anvil-box-{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> DockerSandbox:
        subprocess.run(
            [
                "docker", "run", "-d", "--name", self.name,
                "-v", f"{self.workdir}:/work", "-w", "/work",
                self.image, "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )
        return self

    def exec(self, cmd: str, timeout: float = 120.0) -> tuple[int, str]:
        try:
            r = subprocess.run(
                ["docker", "exec", "-w", "/work", self.name, "sh", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return (124, f"command timed out after {timeout}s")
        return (r.returncode, (r.stdout or "") + (r.stderr or ""))

    def __exit__(self, *exc) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
