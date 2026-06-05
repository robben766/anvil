"""Tests for golden PDF fixtures (D1: PDF fixture generation).

Asserts that the three pre-generated PDFs in packages/kb/golden/pdf/ are
readable and contain the structural traps that D2 (PDF parser) must handle:
  - Company header on every page
  - Markdown table rendered as a real table in 条款篇
  - Key clause text present (等待期为90天)
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import pytest

PDF_DIR = Path(__file__).parent.parent / "golden" / "pdf"

CLAUSE_PDF = PDF_DIR / "01-安康保障计划条款.pdf"
GUIDE_PDF = PDF_DIR / "02-理赔指南.pdf"
PRODUCT_PDF = PDF_DIR / "03-产品说明.pdf"

ALL_PDFS = [CLAUSE_PDF, GUIDE_PDF, PRODUCT_PDF]


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
