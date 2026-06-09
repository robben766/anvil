import json

from anvil_code_agent.eval.swebench import SweInstance, load_instances


def test_load_handles_list_and_jsonstring_fail_to_pass(tmp_path):
    f = tmp_path / "inst.jsonl"
    rows = [
        {"instance_id": "a__b-1", "repo": "a/b", "base_commit": "abc",
         "problem_statement": "fix it", "test_patch": "PATCH",
         "FAIL_TO_PASS": ["t.py::test_x"], "PASS_TO_PASS": []},
        # 官方 HF 格式:FAIL_TO_PASS 是 JSON 字符串
        {"instance_id": "a__b-2", "repo": "a/b", "base_commit": "def",
         "problem_statement": "fix2", "test_patch": "P2",
         "FAIL_TO_PASS": "[\"t.py::test_y\"]", "PASS_TO_PASS": "[]"},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    insts = load_instances(str(f))
    assert len(insts) == 2
    assert isinstance(insts[0], SweInstance)
    assert insts[0].fail_to_pass == ["t.py::test_x"]
    assert insts[1].fail_to_pass == ["t.py::test_y"]  # 字符串也解析成 list
    assert insts[0].repo == "a/b"
    assert insts[0].base_commit == "abc"
