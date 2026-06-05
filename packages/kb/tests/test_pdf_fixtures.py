"""Tests for golden PDF fixtures (D1: PDF fixture generation).

Asserts that the three pre-generated PDFs in packages/kb/golden/pdf/ are
readable and contain the structural traps that D2 (PDF parser) must handle:
  - Company header on every page
  - Markdown table rendered as a real table in 条款篇
  - Key clause text present (等待期为90天)

Also asserts fixture freshness: each PDF's extracted text (whitespace-normalised)
must match the text rendered from the corresponding corpus .md file by
generate_fixtures.generate_pdf().  If a corpus .md is edited without re-running
generate_fixtures.py the freshness test will fail with a descriptive message.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pdfplumber
import pytest

reportlab = pytest.importorskip(
    "reportlab",
    reason="reportlab not installed; skipping fixture freshness tests",
)

PDF_DIR = Path(__file__).parent.parent / "golden" / "pdf"
CORPUS_DIR = Path(__file__).parent.parent / "golden" / "corpus"

CLAUSE_PDF = PDF_DIR / "01-安康保障计划条款.pdf"
GUIDE_PDF = PDF_DIR / "02-理赔指南.pdf"
PRODUCT_PDF = PDF_DIR / "03-产品说明.pdf"

ALL_PDFS = [CLAUSE_PDF, GUIDE_PDF, PRODUCT_PDF]

# ---------------------------------------------------------------------------
# Helpers for fixture freshness
# ---------------------------------------------------------------------------

# Make sure the generate_fixtures module is importable from its non-package path
_FIXTURES_MODULE_DIR = str(PDF_DIR)
if _FIXTURES_MODULE_DIR not in sys.path:
    sys.path.insert(0, _FIXTURES_MODULE_DIR)


def _extract_normalised(pdf_bytes: bytes) -> str:
    """Extract all text from PDF bytes and collapse all whitespace."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        parts = [(page.extract_text() or "") for page in pdf.pages]
    raw = " ".join(parts)
    return re.sub(r"\s+", "", raw)


def _render_pdf_to_bytes(md_path: Path) -> bytes:
    """Render a corpus markdown file to PDF bytes using generate_fixtures."""
    import tempfile  # noqa: PLC0415

    import generate_fixtures  # noqa: PLC0415  (local import, path already patched)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        generate_fixtures.generate_pdf(md_path, tmp_path)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


# Parametrise over the three corpus/PDF pairs
CORPUS_PDF_PAIRS = [
    (CORPUS_DIR / "01-安康保障计划条款.md", CLAUSE_PDF, "条款"),
    (CORPUS_DIR / "02-理赔指南.md", GUIDE_PDF, "理赔指南"),
    (CORPUS_DIR / "03-产品说明.md", PRODUCT_PDF, "产品说明"),
]


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=["条款", "理赔指南", "产品说明"])
def test_pdf_opens(pdf_path: Path) -> None:
    """All three PDFs must exist and open without error."""
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    with pdfplumber.open(pdf_path) as pdf:
        assert len(pdf.pages) >= 1


def test_clause_pdf_at_least_two_pages() -> None:
    """条款篇 must span at least 2 pages (natural content volume)."""
    with pdfplumber.open(CLAUSE_PDF) as pdf:
        assert len(pdf.pages) >= 2, f"Expected ≥2 pages, got {len(pdf.pages)}"


@pytest.mark.parametrize("pdf_path", ALL_PDFS, ids=["条款", "理赔指南", "产品说明"])
def test_every_page_has_company_header(pdf_path: Path) -> None:
    """Each page must contain the company name in the header text."""
    company = "星辉人寿保险股份有限公司"
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            assert company in text, (
                f"{pdf_path.name} page {i + 1} missing company header. "
                f"Text snippet: {text[:200]!r}"
            )


def test_clause_pdf_has_table() -> None:
    """条款篇 must contain at least one extractable table (轻症给付比例表)."""
    found_table = False
    with pdfplumber.open(CLAUSE_PDF) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                found_table = True
                break
    assert found_table, "条款篇 PDF contains no extractable table"


def test_clause_pdf_waiting_period_text() -> None:
    """「等待期为90天」must appear in 条款篇 (whitespace-normalised)."""
    with pdfplumber.open(CLAUSE_PDF) as pdf:
        full_text = " ".join((p.extract_text() or "") for p in pdf.pages)

    normalised = re.sub(r"\s+", "", full_text)
    assert "等待期为90天" in normalised, (
        "Key clause 「等待期为90天」not found in 条款篇 after whitespace normalisation"
    )


# ---------------------------------------------------------------------------
# Fixture freshness: PDF must match the corpus .md it was generated from
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "md_path,pdf_path",
    [(md, pdf) for md, pdf, _ in CORPUS_PDF_PAIRS],
    ids=[label for _, _, label in CORPUS_PDF_PAIRS],
)
def test_pdf_fixture_matches_corpus_md(md_path: Path, pdf_path: Path) -> None:
    """PDF fixture text must match what generate_fixtures.py would produce today.

    Comparison method:
      1. Render the corpus .md to a fresh PDF in a temp file using
         generate_fixtures.generate_pdf().
      2. Extract text from both PDFs with pdfplumber.
      3. Collapse all whitespace in both texts (_extract_normalised).
      4. Assert equality.

    If this test fails it means the corpus .md has been edited but the
    committed PDF has not been regenerated.  Fix: re-run
        uv run python packages/kb/golden/pdf/generate_fixtures.py
    and commit the updated PDFs.
    """
    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    assert md_path.exists(), f"Corpus md not found: {md_path}"

    committed_text = _extract_normalised(pdf_path.read_bytes())
    fresh_text = _extract_normalised(_render_pdf_to_bytes(md_path))

    assert committed_text == fresh_text, (
        f"corpus md 已改,请重跑 generate_fixtures.py\n"
        f"  corpus: {md_path.name}\n"
        f"  pdf:    {pdf_path.name}\n"
        f"  committed chars: {len(committed_text)}, fresh chars: {len(fresh_text)}"
    )
