import inspect

from anvil_code_agent.harness.loop import run, step


def test_step_and_run_accept_summarizer():
    assert "summarizer" in inspect.signature(step).parameters
    assert "summarizer" in inspect.signature(run).parameters
    assert inspect.signature(step).parameters["summarizer"].default is None
