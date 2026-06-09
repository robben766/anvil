import json

from anvil_code_agent.eval.swebench import SweInstance, load_instances


def test_instance_defaults_image_and_install():
    inst = SweInstance(instance_id="a-1", repo="a/b", base_commit="x",
                       problem_statement="p", test_patch="", fail_to_pass=["t::t"])
    assert inst.image == "python:3.11"
    assert inst.install_cmd == ""


def test_load_reads_image_and_install_cmd(tmp_path):
    f = tmp_path / "inst.jsonl"
    f.write_text(json.dumps({
        "instance_id": "a-1", "repo": "a/b", "base_commit": "x",
        "problem_statement": "p", "test_patch": "P",
        "FAIL_TO_PASS": ["t::t"], "PASS_TO_PASS": [],
        "image": "python:3.9", "install_cmd": "pip install -e .",
    }) + "\n")
    inst = load_instances(str(f))[0]
    assert inst.image == "python:3.9"
    assert inst.install_cmd == "pip install -e ."
