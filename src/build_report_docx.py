"""보고서 마크다운(docs/report/cafe_survival_report.md)을 Word 파일로 변환한다.

지원 문법: # ~ ### 제목, 문단, - 글머리표, 1. 번호 목록, | 표 |, ![캡션](그림 경로), **굵게**.
첫 # 제목은 문서 제목, 그 바로 뒤 ## 제목은 부제로 쓴다.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "report" / "cafe_survival_report.md"
OUTPUT = ROOT / "docs" / "report" / "cafe_survival_report.docx"
FONT = "맑은 고딕"
BODY_PT = 10.5
IMAGE_WIDTH_CM = 15.0
PAGE_BREAK_BEFORE = {"요약", "1. 서론", "2. 데이터 및 분석 방법", "3. 분석 결과", "4. 결론", "5. 참고문헌 및 데이터 출처", "부록"}


def set_font(run_or_style, size: float | None = None, bold: bool | None = None, color: RGBColor | None = None) -> None:
    font = run_or_style.font
    font.name = FONT
    element = run_or_style.element if hasattr(run_or_style, "element") else run_or_style._element
    rpr = element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for key in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(key), FONT)
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if color is not None:
        font.color.rgb = color


def setup_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    set_font(normal, BODY_PT)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.3
    heading_sizes = {"Title": 20, "Subtitle": 13, "Heading 1": 15, "Heading 2": 13, "Heading 3": 11.5}
    for name, size in heading_sizes.items():
        style = document.styles[name]
        set_font(style, size, bold=name != "Subtitle", color=RGBColor(0x1F, 0x2A, 0x44))
        style.paragraph_format.space_before = Pt(14 if name.startswith("Heading") else 0)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True
    for name in ("List Bullet", "List Number"):
        set_font(document.styles[name], BODY_PT)


def add_inline(paragraph, text: str) -> None:
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        run = paragraph.add_run(part[2:-2] if bold else part)
        set_font(run, bold=True if bold else None)


def shade(cell, hex_color: str) -> None:
    tc_pr = cell._element.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), hex_color)
    tc_pr.append(shading)


def add_table(document: Document, rows: list[list[str]]) -> None:
    header, body = rows[0], rows[1:]
    table = document.add_table(rows=len(rows), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate([header] + body):
        for c in range(len(header)):
            cell = table.cell(r, c)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.1
            add_inline(paragraph, row[c] if c < len(row) else "")
            for run in paragraph.runs:
                set_font(run, 9, bold=True if r == 0 else None)
            if r == 0:
                shade(cell, "DCE3EE")
    document.add_paragraph()


def parse_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def build() -> None:
    document = Document()
    section = document.sections[0]
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin"):
        setattr(section, side, Cm(2.3))
    section.top_margin = section.bottom_margin = Cm(2.2)
    setup_styles(document)

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title_done = subtitle_done = False
    table_rows: list[list[str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            if not re.fullmatch(r"\|[\s:|-]+\|", line):
                table_rows.append(parse_row(line))
            i += 1
            if i >= len(lines) or not lines[i].startswith("|"):
                add_table(document, table_rows)
                table_rows = []
            continue
        if not line.strip():
            i += 1
            continue
        heading = re.match(r"^(#{1,3}) (.+)$", line)
        image = re.match(r"^!\[(.+)\]\((.+)\)$", line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2)
            if level == 1 and not title_done:
                document.add_paragraph(text, style="Title").alignment = WD_ALIGN_PARAGRAPH.CENTER
                title_done = True
            elif level == 2 and title_done and not subtitle_done:
                document.add_paragraph(text, style="Subtitle").alignment = WD_ALIGN_PARAGRAPH.CENTER
                subtitle_done = True
            else:
                if level == 1 and text in PAGE_BREAK_BEFORE:
                    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
                document.add_heading(text, level=level)
        elif image:
            caption, relative = image.groups()
            path = (SOURCE.parent / relative).resolve()
            picture_paragraph = document.add_paragraph()
            picture_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            picture_paragraph.paragraph_format.keep_with_next = True
            picture_paragraph.add_run().add_picture(str(path), width=Cm(IMAGE_WIDTH_CM))
            caption_paragraph = document.add_paragraph()
            caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(caption_paragraph, caption)
            for run in caption_paragraph.runs:
                set_font(run, 9)
                run.font.italic = True
        elif re.match(r"^- ", line):
            add_inline(document.add_paragraph(style="List Bullet"), line[2:])
        elif re.match(r"^\d+\. ", line):
            # Word 자동 번호는 문서 전체에서 이어지므로 번호를 글자로 넣는다.
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.63)
            paragraph.paragraph_format.first_line_indent = Cm(-0.63)
            add_inline(paragraph, line)
        else:
            paragraph = document.add_paragraph()
            if not subtitle_done or (title_done and len(document.paragraphs) <= 3):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, line)
        i += 1

    document.save(OUTPUT)
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    build()
