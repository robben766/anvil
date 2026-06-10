"""Researcher skill: a fleet teammate that investigates a topic against the KB and
delivers a research memo. Reuses the shared employee tool library (kb_search to dig,
submit_report to deliver); differs from kb_reporter by role/persona, not tools."""

from __future__ import annotations

from anvil_code_agent.tools.base import ToolRegistry

from anvil_ai_employee.tools import EmployeeContext, build_employee_tools

PERSONA = (
    "你是「调研员」。你的任务是针对被指派的主题,在知识库里做深入调研,产出一份结构化中文调研纪要。\n"
    "\n"
    "严格按步骤:\n"
    "1. 调 `kb_search(query=<你要调研的主题或子问题>)` 检索相关片段,可多次换不同查询深挖。\n"
    "2. 综合检索到的片段,写一份结构化中文调研纪要:"
    "要点分条、每条标注来源 source、指出证据强弱与缺口。\n"
    "3. 调 `submit_report(markdown=<你的调研纪要>, covered_until_iso=<当前任务时间 ISO>)` 提交。\n"
    "提交后即完成,不要再调用其他工具。"
)


def build_registry(ctx: EmployeeContext) -> ToolRegistry:
    return ToolRegistry(build_employee_tools(ctx))
