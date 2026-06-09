from anvil_code_agent.repomap.tags import extract_tags


def test_extract_defs_and_refs():
    code = (
        "def add(a, b):\n"
        "    return helper(a) + b\n"
        "\n"
        "class Calc:\n"
        "    def run(self):\n"
        "        return add(1, 2)\n"
        "\n"
        "def helper(x):\n"
        "    return x\n"
    )
    tags = extract_tags(code)
    assert tags.defs == {"add", "Calc", "run", "helper"}
    # 引用:helper、add 被调用(self/return 等不是 identifier-call)
    assert "helper" in tags.refs
    assert "add" in tags.refs


def test_empty_and_syntax_tolerant():
    tags = extract_tags("")
    assert tags.defs == set()
    assert tags.refs == []
    # 语法不完整也不崩(tree-sitter 容错)
    tags2 = extract_tags("def broken(:\n    x =")
    assert isinstance(tags2.defs, set)
