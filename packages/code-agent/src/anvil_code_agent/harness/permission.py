"""Tool risk tiers + approval policies. A policy is a callable
(name, args, risk) -> bool; returning False blocks the tool (the loop turns the block
into model feedback). Default policy in eval is auto_approve; interactive callers can
swap in a gate that requires human approval for high-risk tools."""

from __future__ import annotations

from typing import Any, Callable

ApprovalPolicy = Callable[[str, dict[str, Any], str], bool]

_RISK = {
    "read_file": "low",
    "grep": "low",
    "repo_map": "low",
    "edit_file": "medium",
    "run_tests": "medium",
    "bash": "high",
}
_ORDER = {"low": 0, "medium": 1, "high": 2}


def risk_level(name: str) -> str:
    """Unknown tools default to 'high' — fail safe."""
    return _RISK.get(name, "high")


def auto_approve(name: str, args: dict[str, Any], risk: str) -> bool:
    return True


def deny_high(name: str, args: dict[str, Any], risk: str) -> bool:
    return risk != "high"


def gate_by_risk(max_risk: str) -> ApprovalPolicy:
    """Allow tools whose risk is <= max_risk; block the rest."""
    ceiling = _ORDER[max_risk]

    def policy(name: str, args: dict[str, Any], risk: str) -> bool:
        return _ORDER.get(risk, 2) <= ceiling

    return policy
