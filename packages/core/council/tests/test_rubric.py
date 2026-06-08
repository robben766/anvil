from anvil_council.rubric import DEFAULT_RUBRIC, Criterion, Rubric


def test_default_rubric_has_functional_criteria():
    assert isinstance(DEFAULT_RUBRIC, Rubric)
    keys = DEFAULT_RUBRIC.keys()
    assert keys == ["correctness", "evidence", "completeness", "relevance"]
    for c in DEFAULT_RUBRIC.criteria:
        assert isinstance(c, Criterion)
        assert c.description


def test_rubric_keys_helper():
    r = Rubric(name="t", criteria=(Criterion("a", "desc a"), Criterion("b", "desc b")))
    assert r.keys() == ["a", "b"]
