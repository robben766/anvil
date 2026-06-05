"""
PDF fixture generator for anvil-kb golden corpus.

Usage:
    uv run python packages/kb/golden/pdf/generate_fixtures.py

Reads packages/kb/golden/corpus/*.md and renders them into PDF files
in the same directory as this script. Each PDF includes:
  - Page header: 「星辉人寿保险股份有限公司」(9pt, centered)
  - Page footer: 「第 N 页」(9pt, centered)
  - Markdown headings rendered at 18/15/12pt
  - Body text at 10.5pt
  - Markdown tables rendered as ReportLab Table with grid lines

Chinese font: STSong-Light (built-in CID font, no system font dependency)
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Font registration
# ---------------------------------------------------------------------------
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

FONT = "STSong-Light"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
H1 = ParagraphStyle("H1", fontName=FONT, fontSize=18, leading=26, spaceAfter=8)
H2 = ParagraphStyle("H2", fontName=FONT, fontSize=15, leading=22, spaceAfter=6, spaceBefore=10)
H3 = ParagraphStyle("H3", fontName=FONT, fontSize=12, leading=18, spaceAfter=4, spaceBefore=6)
BODY = ParagraphStyle("Body", fontName=FONT, fontSize=10.5, leading=16, spaceAfter=4)

# ---------------------------------------------------------------------------
# Header / footer callback
# ---------------------------------------------------------------------------
PAGE_WIDTH, PAGE_HEIGHT = A4


def _draw_header_footer(canvas, doc):
    """Draw header and footer on every page."""
    canvas.saveState()
    canvas.setFont(FONT, 9)
    # Header – centered, near top
    canvas.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 15 * mm, "星辉人寿保险股份有限公司")
    # Footer – centered, near bottom
    page_num = doc.page
    canvas.drawCentredString(PAGE_WIDTH / 2, 10 * mm, f"第 {page_num} 页")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Markdown → ReportLab flowables
# ---------------------------------------------------------------------------

def _parse_table(lines: list[str]) -> Table:
    """Convert markdown table lines into a ReportLab Table."""
    rows: list[list[str]] = []
    for line in lines:
        # Skip separator rows (---|---|...)
        if re.match(r"^\|[-: |]+\|$", line.strip()):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return Table([[""]])

    # Build paragraph-based cells for proper font rendering
    cell_data = []
    for _i, row in enumerate(rows):
        style = ParagraphStyle(
            "TableCell",
            fontName=FONT,
            fontSize=9,
            leading=13,
        )
        cell_data.append([Paragraph(cell, style) for cell in row])

    col_count = max(len(r) for r in cell_data)
    page_inner_width = PAGE_WIDTH - 40 * mm  # margins ~20mm each side
    col_width = page_inner_width / col_count

    tbl = Table(cell_data, colWidths=[col_width] * col_count)
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def md_to_flowables(md_text: str) -> list:
    """Lightly parse markdown text into ReportLab flowables."""
    flowables: list = []
    lines = md_text.splitlines()

    # Collect table lines as a block
    table_buffer: list[str] = []

    def flush_table():
        if table_buffer:
            tbl = _parse_table(table_buffer)
            flowables.append(Spacer(1, 4))
            flowables.append(tbl)
            flowables.append(Spacer(1, 4))
            table_buffer.clear()

    for line in lines:
        stripped = line.strip()

        # Detect markdown table line (starts and ends with |)
        if stripped.startswith("|"):
            table_buffer.append(stripped)
            continue
        else:
            flush_table()

        if not stripped:
            flowables.append(Spacer(1, 6))
            continue

        if stripped.startswith("### "):
            text = stripped[4:].strip()
            flowables.append(Paragraph(text, H3))
        elif stripped.startswith("## "):
            text = stripped[3:].strip()
            flowables.append(Paragraph(text, H2))
        elif stripped.startswith("# "):
            text = stripped[2:].strip()
            flowables.append(Paragraph(text, H1))
        else:
            # Escape special chars that ReportLab XML parser chokes on
            safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Strip bold markers (**text**) → just text (minimal md support)
            safe = re.sub(r"\*\*(.+?)\*\*", r"\1", safe)
            flowables.append(Paragraph(safe, BODY))

    flush_table()
    return flowables


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
CORPUS_DIR = HERE.parent / "corpus"

CORPUS_FILES = [
    "01-安康保障计划条款.md",
    "02-理赔指南.md",
    "03-产品说明.md",
]


def generate_pdf(md_path: Path, out_path: Path) -> None:
    """Generate a PDF from a markdown file."""
    md_text = md_path.read_text(encoding="utf-8")

    # Margins: 20mm each side, 25mm top/bottom to leave room for header/footer
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
    )

    flowables = md_to_flowables(md_text)

    # For 条款篇 (doc 01) ensure ≥2 pages by inserting extra spacing if needed.
    # We use a generous leading / spacer approach; if the content is already
    # long enough this has no effect.  We do NOT add fake content.
    if "条款" in md_path.name:
        # Insert large spacers between top-level sections so content flows
        # naturally across page boundaries.
        extra: list = []
        for item in flowables:
            extra.append(item)
            if isinstance(item, Paragraph) and item.style == H2:
                extra.append(Spacer(1, 12))
        flowables = extra

    doc.build(flowables, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    print(f"Generated: {out_path} ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    for filename in CORPUS_FILES:
        md_path = CORPUS_DIR / filename
        stem = md_path.stem  # e.g. "01-安康保障计划条款"
        out_path = HERE / f"{stem}.pdf"
        generate_pdf(md_path, out_path)
    print("Done. All 3 PDFs generated.")


if __name__ == "__main__":
    main()
