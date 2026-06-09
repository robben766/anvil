"""Skills tier: versioned markdown persona/skill files loaded at runtime (the third tier
of the three-tier memory). M1/M2 hard-coded personas in Python; these externalize them."""

from __future__ import annotations

from pathlib import Path

_SKILLS_DIR = Path(__file__).parent / "skills"


def load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"skill not found: {name} (looked in {_SKILLS_DIR})")
    return path.read_text(encoding="utf-8").strip()


def available_skills() -> list[str]:
    return sorted(p.stem for p in _SKILLS_DIR.glob("*.md"))
