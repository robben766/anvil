from anvil_code_agent.cli import build_parser


def test_parser_has_solve_and_eval():
    p = build_parser()
    ns = p.parse_args(["eval", "--dataset", "tasks.jsonl", "--model", "deepseek-chat"])
    assert ns.command == "eval"
    assert ns.dataset == "tasks.jsonl"
    assert ns.model == "deepseek-chat"


def test_parser_solve():
    p = build_parser()
    ns = p.parse_args(["solve", "--repo", "/tmp/r", "--prompt", "fix it"])
    assert ns.command == "solve"
    assert ns.repo == "/tmp/r"
    assert ns.prompt == "fix it"
