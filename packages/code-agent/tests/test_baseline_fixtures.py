import shutil
import subprocess
from pathlib import Path

from anvil_code_agent.eval.task import load_tasks

GOLDEN = Path(__file__).parent.parent / "src" / "anvil_code_agent" / "eval" / "golden"


def test_baseline_jsonl_loads_multifile_tasks():
    tasks = load_tasks(str(GOLDEN / "baseline.jsonl"))
    ids = {t.id for t in tasks}
    assert {"strops-upper", "counter-evens"} <= ids


def _fails_before_passes_after(tmp_path, fixture, bug, fix):
    dst = tmp_path / fixture
    shutil.copytree(GOLDEN / "fixtures" / fixture, dst)
    before = subprocess.run(["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True)
    assert before.returncode != 0  # 带 bug → 失败
    # 在所有 .py 里把 bug 改成 fix
    for py in dst.rglob("*.py"):
        txt = py.read_text()
        if bug in txt:
            py.write_text(txt.replace(bug, fix))
    after = subprocess.run(["python", "-m", "pytest", "-q"], cwd=dst, capture_output=True, text=True)
    assert after.returncode == 0  # 修后 → 通过


def test_strops_fixture_wellformed(tmp_path):
    _fails_before_passes_after(tmp_path, "strops", "s.lower()", "s.upper()")


def test_counter_fixture_wellformed(tmp_path):
    _fails_before_passes_after(tmp_path, "counter", "n % 2 == 1", "n % 2 == 0")
