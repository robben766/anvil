from anvil_code_agent.cli import build_parser


def test_parser_has_swebench():
    p = build_parser()
    ns = p.parse_args(["swebench", "--dataset", "inst.jsonl", "--limit", "3", "--workdir", "/tmp/wb"])
    assert ns.command == "swebench"
    assert ns.dataset == "inst.jsonl"
    assert ns.limit == 3
    assert ns.workdir == "/tmp/wb"
