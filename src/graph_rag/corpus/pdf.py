"""Render documents to born-digital PDFs.

OCR is out of scope, so a generator that rasterised text would silently break every
downstream ticket: the PDFs must carry a real text layer. ReportLab draws text as text.

`invariant=1` and a fixed creation date keep the bytes identical across runs for the same
seed, so a demo is reproducible down to the file.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from graph_rag.corpus.text import Document

# Fixed so repeated generation produces byte-identical files.
FIXED_TIMESTAMP = "D:20240101120000+00'00'"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CorpusTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18,
            leading=22, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "CorpusSubtitle", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=11, leading=14, spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "CorpusHeading", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, spaceBefore=12, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "CorpusBody", parent=base["BodyText"], fontName="Helvetica", fontSize=10,
            leading=14, alignment=TA_JUSTIFY, spaceAfter=8,
        ),
        "meta": ParagraphStyle(
            "CorpusMeta", parent=base["Normal"], fontName="Helvetica", fontSize=9,
            leading=12,
        ),
    }


def _metadata_table(document: Document, styles) -> Table:
    rows = [
        [Paragraph(f"<b>{key}</b>", styles["meta"]), Paragraph(value, styles["meta"])]
        for key, value in document.metadata
    ]
    table = Table(rows, colWidths=(45 * mm, 110 * mm), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _section_table(rows, styles) -> Table:
    data = [[Paragraph(cell, styles["meta"]) for cell in row] for row in rows]
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, "#999999"),
                ("BACKGROUND", (0, 0), (-1, 0), "#e8e8e8"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def render(document: Document, destination: Path) -> Path:
    """Write one document to `destination` and return the path."""
    styles = _styles()
    template = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        title=document.title,
        author="Vektor Consulting GmbH",
        subject=document.subtitle,
        invariant=1,
    )

    story = [
        Paragraph(document.title, styles["title"]),
        Paragraph(document.subtitle, styles["subtitle"]),
        _metadata_table(document, styles),
        Spacer(1, 8 * mm),
    ]
    for section in document.sections:
        block = [Paragraph(section.heading, styles["heading"])]
        block.extend(Paragraph(text, styles["body"]) for text in section.paragraphs)
        if section.table:
            block.append(Spacer(1, 2 * mm))
            block.append(_section_table(section.table, styles))
            block.append(Spacer(1, 4 * mm))
        # Keep a heading with its first paragraph; let the rest flow.
        story.append(KeepTogether(block[:2]))
        story.extend(block[2:])

    destination.parent.mkdir(parents=True, exist_ok=True)
    template.build(story)
    _freeze_timestamps(destination)
    return destination


def _freeze_timestamps(path: Path) -> None:
    """ReportLab stamps a creation date even under `invariant`; pin it so repeated
    generation with the same seed produces identical bytes."""
    raw = path.read_bytes()
    frozen = re.sub(rb"D:\d{14}[+\-]\d{2}'\d{2}'", FIXED_TIMESTAMP.encode(), raw)
    if frozen != raw:
        path.write_bytes(frozen)
