import os
import shutil
import subprocess
from pathlib import Path

import pytest
from anvil_code_agent.eval.runner import solve_task
from anvil_code_agent.eval.task import Task

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def _init_repo(p):
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "init"],
    ):
        subprocess.run(cmd, cwd=p, check=True)


@pytest.mark.live
async def test_real_agent_fixes_calc_bug(tmp_path):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    db_url = os.environ.get("ANVIL_DATABASE_URL")
    if not api_key or not db_url:
        pytest.skip("needs DEEPSEEK_API_KEY and ANVIL_DATABASE_URL")
    from anvil_gateway import configure

    configure(database_url=os.environ["ANVIL_DATABASE_URL"])
    repo = tmp_path / "calc"
    shutil.copytree(GOLDEN / "fixtures" / "calc", repo)
    _init_repo(repo)
    from anvil_code_agent.eval.task import load_tasks

    real = load_tasks(str(GOLDEN / "tasks.jsonl"))[0]
    task = Task(id=real.id, repo=str(repo), prompt=real.prompt, verify_cmd=real.verify_cmd)
    res = await solve_task(task, model="deepseek-chat", max_steps=15)
    assert res.passed is True
