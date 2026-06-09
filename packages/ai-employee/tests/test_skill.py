from anvil_ai_employee.skills.kb_digest import PERSONA, build_registry
from anvil_ai_employee.tools import EmployeeContext


def test_persona_mentions_steps():
    assert "recall_marker" in PERSONA
    assert "submit_report" in PERSONA


def test_build_registry_has_four_tools():
    ctx = EmployeeContext(session_factory=None, employee="kb_reporter", job_id=None)
    reg = build_registry(ctx)
    names = {sch["function"]["name"] for sch in reg.schemas()}
    assert names == {"recall_marker", "kb_recent", "kb_search", "submit_report"}
