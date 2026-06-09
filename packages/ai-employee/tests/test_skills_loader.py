from anvil_ai_employee.skills_loader import available_skills, load_skill


def test_load_existing_skill():
    text = load_skill("assistant")
    assert isinstance(text, str) and len(text) > 0


def test_unknown_skill_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_skill("nope")


def test_available_lists_md_files():
    skills = available_skills()
    assert "assistant" in skills
