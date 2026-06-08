import shutil
import subprocess
from pathlib import Path

from anvil_code_agent.eval.task import load_tasks

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def test_tasks_jsonl_loads():
    tasks = load_tasks(str(GOLDEN / "tasks.jsonl"))
    assert len(tasks) >= 1
    assert any(t.id == "calc-add" for t in tasks)


def test_fixture_test_fails_on_buggy_code(tmp_path):
    # 把 buggy fixture 拷出来,确认它的测试现在就是 FAIL 的(否则任务没意义)
    src = GOLDEN / "fixtures" / "calc"
    dst = tmp_path / "calc"
    shutil.copytree(src, dst)
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True
    )
    assert proc.returncode != 0  # buggy → 测试失败


def test_fixture_passes_after_correct_fix(tmp_path):
    # 应用正确修复后测试应转绿,证明任务可解
    src = GOLDEN / "fixtures" / "calc"
    dst = tmp_path / "calc"
    shutil.copytree(src, dst)
    code = (dst / "calc.py").read_text().replace("a - b", "a + b")
    (dst / "calc.py").write_text(code)
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True
    )
    assert proc.returncode == 0
