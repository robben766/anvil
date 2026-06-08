import json

from anvil_code_agent.eval.task import Task, load_tasks


def test_task_fields():
    t = Task(id="t1", repo="fixtures/calc", prompt="fix add", verify_cmd="python -m pytest -q")
    assert t.id == "t1"
    assert t.verify_cmd == "python -m pytest -q"


def test_load_tasks_from_jsonl(tmp_path):
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        json.dumps({"id": "t1", "repo": "fixtures/calc", "prompt": "fix", "verify_cmd": "pytest"})
        + "\n"
    )
    tasks = load_tasks(str(f))
    assert len(tasks) == 1
    assert isinstance(tasks[0], Task)
    assert tasks[0].id == "t1"


def test_load_tasks_resolves_relative_repo_against_dataset_dir(tmp_path):
    """Relative repo paths in jsonl must be resolved absolute, relative to dataset dir."""
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        json.dumps({"id": "t1", "repo": "fixtures/x", "prompt": "fix", "verify_cmd": "pytest"})
        + "\n"
    )
    tasks = load_tasks(str(f))
    assert tasks[0].repo == str(tmp_path / "fixtures" / "x")


def test_load_tasks_keeps_absolute_repo(tmp_path):
    """Absolute repo paths must be stored unchanged."""
    abs_repo = str(tmp_path / "abs" / "repo")
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        json.dumps({"id": "t1", "repo": abs_repo, "prompt": "fix", "verify_cmd": "pytest"})
        + "\n"
    )
    tasks = load_tasks(str(f))
    assert tasks[0].repo == abs_repo
