"""Prompt-injection detection.

Fast path: deterministic bilingual keyword/regex matching — zero latency, no LLM.
Optional semantic fallback (detect_injection_llm) lives below, default OFF.

Design: each pattern carries a category. A match flags injection with confidence
scaled by how many distinct patterns fired (capped at 1.0). Benign queries that merely
contain words like "忽略"/"ignore"/"system" do not match because patterns require the
*adversarial collocation* (e.g. ignore + instructions), not the lone word.

Deliberate limits (delegated to detect_injection_llm, the optional semantic fallback):
obfuscated/leetspeak attacks (e.g. "1gn0re"), non-EN/ZH languages, and novel paraphrases
that don't reuse the enumerated vocabulary will slip past this deterministic fast path.
The fast path optimizes for sub-millisecond rejection of common, literal attacks at high
precision (rarely blocking legitimate users), NOT for exhaustive recall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InjectionVerdict:
    is_injection: bool
    category: str  # the category of the first/strongest match, or "none"
    matched: list[str] = field(default_factory=list)  # names of patterns that fired
    confidence: float = 0.0


# (pattern_name, category, compiled_regex). Order matters: earlier categories win ties.
_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "zh_ignore_instructions",
        "instruction_override",
        re.compile(r"(忽略|无视|不要管|不用管|不要理会|别理会)[^。\n]{0,12}(指令|提示|规则|要求|设定|命令)"),
    ),
    (
        "zh_new_rules_override",
        "instruction_override",
        re.compile(r"(新的?规则|从现在起|现在开始)[^。\n]{0,16}(无视|忽略|不受限制|没有任何限制)"),
    ),
    (
        "en_ignore_instructions",
        "instruction_override",
        re.compile(
            r"(ignore|disregard|forget)\s+(all\s+|the\s+)?(previous|above|prior|earlier)?\s*"
            r"(instructions|prompts|rules|everything|directions|commands)",
            re.IGNORECASE,
        ),
    ),
    (
        "zh_prompt_leak",
        "prompt_leak",
        re.compile(
            r"(输出|显示|泄露|发给我|重复|复述|告诉我).{0,12}(系统提示|提示词|初始指令|你的指令)"
            r"|(系统提示|提示词|初始指令|你的指令).{0,12}(复述|输出|发出来|发给我|说一遍)"
        ),
    ),
    (
        "en_prompt_leak",
        "prompt_leak",
        re.compile(
            r"(reveal|print|repeat|show|output|leak).{0,20}"
            r"(system prompt|initial instructions|hidden (config|configuration|prompt)"
            r"|the (words|text|prompt) above)",
            re.IGNORECASE,
        ),
    ),
    (
        "zh_jailbreak_roleplay",
        "jailbreak",
        re.compile(r"(扮演|假装你是|你现在是|你是一个).{0,16}(没有.{0,4}限制|黑客|不受约束|开发者模式)"),
    ),
    (
        "en_jailbreak_roleplay",
        "jailbreak",
        re.compile(
            r"(you are now|pretend you are|developer mode|\bDAN\b)"
            r"|act as\s+(an?\s+)?(unrestricted|jailbroken|evil|uncensored|developer|dan)"
            r"|(unrestricted|jailbroken|no restrictions|restrictions are disabled)",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        "delimiter_injection",
        re.compile(
            r"(```|<\|)\s*(system|inst)|\[INST\]|\[/INST\]|(^|\n)\s*system\s*:"
            r"|(override|disable).{0,12}(guardrail|safety)",
            re.IGNORECASE,
        ),
    ),
]


def detect_injection(text: str) -> InjectionVerdict:
    """Deterministic rule-based injection check. Pure function, zero network."""
    if not text or not text.strip():
        return InjectionVerdict(is_injection=False, category="none")
    matched: list[str] = []
    first_category = "none"
    for name, category, pattern in _PATTERNS:
        if pattern.search(text):
            matched.append(name)
            if first_category == "none":
                first_category = category
    if not matched:
        return InjectionVerdict(is_injection=False, category="none")
    confidence = min(1.0, 0.6 + 0.2 * (len(matched) - 1))
    return InjectionVerdict(
        is_injection=True,
        category=first_category,
        matched=matched,
        confidence=confidence,
    )
