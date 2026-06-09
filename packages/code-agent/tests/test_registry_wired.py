from anvil_code_agent.eval.runner import default_registry


def test_default_registry_includes_search_tools():
    names = [s["function"]["name"] for s in default_registry().schemas()]
    assert "read_file" in names
    assert "edit_file" in names
    assert "bash" in names
    assert "run_tests" in names
    assert "repo_map" in names
    assert "grep" in names
