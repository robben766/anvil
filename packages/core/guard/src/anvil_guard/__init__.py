"""anvil-guard: universal security guardrails (injection detection + structured output)."""

from anvil_guard.injection import InjectionVerdict, detect_injection
from anvil_guard.structured import StructuredOutputError, structured_chat

__version__ = "0.1.0"
__all__ = [
    "InjectionVerdict",
    "StructuredOutputError",
    "detect_injection",
    "structured_chat",
]
