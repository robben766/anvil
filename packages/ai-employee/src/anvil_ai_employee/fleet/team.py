"""The roster of employees a fleet can dispatch to. Each Employee bundles a persona, a
registry builder (its ACI), and a default task prompt. The supervisor reads `description`
to decide who gets which subtask."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.skills import kb_digest, research
from anvil_ai_employee.tools import EmployeeContext


@dataclass
class Employee:
    name: str
    description: str
    persona: str
    build_registry: Callable[[EmployeeContext], ToolRegistry]
    task_prompt: str = "现在开始执行你被指派的任务。"


EMPLOYEES: dict[str, Employee] = {
    "kb_reporter": Employee(
        name="kb_reporter",
        description="知识库周报员:汇总知识库新增内容,产结构化中文周报。",
        persona=kb_digest.PERSONA,
        build_registry=kb_digest.build_registry,
        task_prompt="现在开始产出本期知识库周报。",
    ),
    "researcher": Employee(
        name="researcher",
        description="调研员:针对某主题在知识库做深入语义检索,产结构化调研纪要。",
        persona=research.PERSONA,
        build_registry=research.build_registry,
        task_prompt="现在开始针对你被指派的主题做调研。",
    ),
}
