from anvil_code_agent.harness.permission import (
    auto_approve,
    deny_high,
    gate_by_risk,
    risk_level,
)


def test_risk_levels():
    assert risk_level("read_file") == "low"
    assert risk_level("grep") == "low"
    assert risk_level("edit_file") == "medium"
    assert risk_level("bash") == "high"
    assert risk_level("unknown_tool") == "high"  # 未知工具按最高风险(安全默认)


def test_auto_approve_allows_all():
    assert auto_approve("bash", {"cmd": "rm -rf /"}, "high") is True


def test_deny_high_blocks_only_high():
    assert deny_high("read_file", {}, "low") is True
    assert deny_high("edit_file", {}, "medium") is True
    assert deny_high("bash", {}, "high") is False


def test_gate_by_risk_factory():
    only_low = gate_by_risk("low")  # 只放行 <= low
    assert only_low("read_file", {}, "low") is True
    assert only_low("edit_file", {}, "medium") is False
    up_to_medium = gate_by_risk("medium")
    assert up_to_medium("edit_file", {}, "medium") is True
    assert up_to_medium("bash", {}, "high") is False
