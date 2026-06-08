from pathlib import Path

from anvil_eval.dataset import load_dataset

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "kb.jsonl"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "golden" / "corpus"


def _corpus_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md")))


def test_dataset_loads_and_is_large_enough():
    cases = load_dataset(str(GOLDEN))
    assert len(cases) >= 50, f"expected >=50 golden cases, got {len(cases)}"


def test_ids_unique():
    cases = load_dataset(str(GOLDEN))
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_answerable_cases_have_grounded_evidences():
    corpus = _corpus_text()
    cases = load_dataset(str(GOLDEN))
    answerable = [c for c in cases if c.answerable]
    assert len(answerable) >= 40
    for c in answerable:
        assert c.evidences, f"{c.id}: answerable case must have evidences"
        for ev in c.evidences:
            assert ev in corpus, f"{c.id}: evidence not found verbatim in corpus: {ev!r}"


def test_has_refusal_cases():
    cases = load_dataset(str(GOLDEN))
    refusals = [c for c in cases if not c.answerable]
    assert len(refusals) >= 5, "need >=5 unanswerable/refusal cases for the refusal axis"


def test_refusal_cases_have_empty_evidences():
    # A refusal whose "evidence" is actually in the corpus would be a mislabeled case.
    cases = load_dataset(str(GOLDEN))
    for c in cases:
        if not c.answerable:
            assert not c.evidences, f"{c.id}: refusal case must have empty evidences"
