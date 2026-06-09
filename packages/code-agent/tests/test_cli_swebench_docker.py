from anvil_code_agent.cli import build_parser


def test_swebench_parser_has_docker_flag():
    p = build_parser()
    ns = p.parse_args(["swebench", "--dataset", "i.jsonl", "--docker"])
    assert ns.command == "swebench"
    assert ns.docker is True


def test_swebench_docker_defaults_false():
    p = build_parser()
    ns = p.parse_args(["swebench", "--dataset", "i.jsonl"])
    assert ns.docker is False
