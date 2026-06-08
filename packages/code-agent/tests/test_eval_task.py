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
