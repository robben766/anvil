def test_employees_registry_has_two_distinct_employees():
    from anvil_ai_employee.fleet.team import EMPLOYEES

    assert set(EMPLOYEES) >= {"kb_reporter", "researcher"}
    kb = EMPLOYEES["kb_reporter"]
    rs = EMPLOYEES["researcher"]
    assert kb.name == "kb_reporter"
    assert rs.name == "researcher"
    assert kb.persona != rs.persona  # distinct roles
    assert kb.description and rs.description  # supervisor needs capability blurbs


def test_employee_build_registry_returns_toolregistry(session_factory):
    from anvil_ai_employee.fleet.team import EMPLOYEES
    from anvil_ai_employee.tools import EmployeeContext
    from anvil_code_agent.tools.base import ToolRegistry

    ctx = EmployeeContext(session_factory=session_factory, employee="researcher", job_id=None)
    reg = EMPLOYEES["researcher"].build_registry(ctx)
    assert isinstance(reg, ToolRegistry)
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "kb_search" in names
    assert "submit_report" in names
