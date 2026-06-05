"""防腐烂测试:验证 packages/kb/golden/kb.jsonl 与 corpus 语料保持一致。

运行方式: uv run pytest packages/kb -q  (从仓库根执行)
"""
from __future__ import annotations

import pathlib

import pytest
from anvil_eval.dataset import load_dataset

# ── 路径 ─────────────────────────────────────────────────────────────────────
# __file__ = packages/kb/tests/test_golden_dataset.py
# REPO_ROOT = packages/kb/../../..  → 仓库根
_KB_ROOT = pathlib.Path(__file__).parent.parent
_GOLDEN_JSONL = _KB_ROOT / "golden" / "kb.jsonl"
_CORPUS_DIR = _KB_ROOT / "golden" / "corpus"


def _load_corpus_texts() -> list[str]:
    """读取 corpus/ 下所有 .md 文件并返回其文本列表。"""
    return [f.read_text(encoding="utf-8") for f in sorted(_CORPUS_DIR.glob("*.md"))]


# ── 基础加载测试 ──────────────────────────────────────────────────────────────


def test_load_dataset_succeeds_with_12_cases():
    """kb.jsonl 能被 load_dataset 加载且恰好有 12 条用例。"""
    cases = load_dataset(str(_GOLDEN_JSONL))
    assert len(cases) == 12, f"Expected 12 cases, got {len(cases)}"


# ── evidence 子串验证 ─────────────────────────────────────────────────────────


def test_all_evidences_are_corpus_substrings():
    """每条 evidence 空白归一化后必须是 corpus 至少一篇文本的子串。"""
    cases = load_dataset(str(_GOLDEN_JSONL))
    corpus_texts = _load_corpus_texts()
    normalized_corpus = ["".join(t.split()) for t in corpus_texts]

    failures: list[str] = []
    for case in cases:
        for ev in case.evidences:
            normalized_ev = "".join(ev.split())
            found = any(normalized_ev in nc for nc in normalized_corpus)
            if not found:
                failures.append(f"[{case.id}] evidence not found in corpus: {ev!r}")

    assert not failures, "Evidence substring check failed:\n" + "\n".join(failures)


# ── answerable 字段一致性 ─────────────────────────────────────────────────────


def test_answerable_false_cases_have_empty_evidences():
    """answerable=false 的用例 evidences 必须为空列表。"""
    cases = load_dataset(str(_GOLDEN_JSONL))
    violations = [
        f"{c.id}: answerable=false but evidences={c.evidences!r}"
        for c in cases
        if not c.answerable and c.evidences
    ]
    assert not violations, "\n".join(violations)


def test_answerable_true_cases_have_nonempty_evidences():
    """answerable=true 的用例 evidences 必须非空。"""
    cases = load_dataset(str(_GOLDEN_JSONL))
    violations = [
        f"{c.id}: answerable=true but evidences is empty"
        for c in cases
        if c.answerable and not c.evidences
    ]
    assert not violations, "\n".join(violations)


# ── 参数化逐条验证 ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def kb_cases():
    return load_dataset(str(_GOLDEN_JSONL))


@pytest.fixture(scope="module")
def normalized_corpus():
    return ["".join(t.split()) for t in _load_corpus_texts()]


@pytest.mark.parametrize(
    "case_id",
    [f"kb-{i:02d}" for i in range(1, 13)],
)
def test_case_exists(case_id, kb_cases):
    """每个预期 id 在数据集中都存在。"""
    ids = {c.id for c in kb_cases}
    assert case_id in ids, f"Missing case id: {case_id}"
