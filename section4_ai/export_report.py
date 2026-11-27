"""Export the Section 4 Markdown source, benchmark table and figure to editable Word."""

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def set_style(style, font, size, before=0, after=0, bold=False):
    style.font.name = font
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor.from_string('000000')
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)


def add_inline(paragraph, text):
    for index, chunk in enumerate(re.split(r'\*([^*]+)\*', text)):
        paragraph.add_run(chunk).italic = bool(index % 2)
    return paragraph


def append_table(document, lines):
    cells = [[value.strip() for value in line.strip('|').split('|')] for line in lines]
    data = [row for row in cells if not all(re.fullmatch(r':?-+:?', value) for value in row)]
    table = document.add_table(rows=len(data), cols=len(data[0]))
    widths = (1.55, .69, .62, .72, .64, .67, .65, .73)
    table.autofit = False
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Inches(width)
    for row_index, (row, values) in enumerate(zip(table.rows, data)):
        for cell_index, (cell, text) in enumerate(zip(row.cells, values)):
            cell.width = Inches(widths[cell_index])
            paragraph = cell.paragraphs[0]
            paragraph.style = document.styles['Benchmark Cell']
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT if cell_index else WD_ALIGN_PARAGRAPH.LEFT
            run = paragraph.add_run(text)
            run.bold = row_index == 0
            if row_index == 0 or row_index % 2 == 0:
                shading = OxmlElement('w:shd')
                shading.set(qn('w:fill'), 'E6EEF3' if row_index == 0 else 'F2F6F8')
                cell._tc.get_or_add_tcPr().append(shading)
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement('w:cantSplit'))
        if row_index == 0:
            tr_pr.append(OxmlElement('w:tblHeader'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='?', default=Path(__file__).with_name('Section4_AI_Integration.md'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    document = Document()
    document.core_properties.title = 'Section 4 AI Integration'
    document.core_properties.subject = 'ROSE5770 Cooperative Multi-UAV Hill Fire Detection'
    document.core_properties.author = 'ROSE5770 Project Team'
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.left_margin = section.right_margin = Inches(.75)
    section.top_margin = section.bottom_margin = Inches(.72)
    styles = document.styles
    set_style(styles['Normal'], 'Calibri', 11, after=7)
    styles['Normal'].paragraph_format.line_spacing = 1.13
    set_style(styles['Title'], 'Calibri', 23, after=16, bold=True)
    set_style(styles['Heading 1'], 'Calibri', 13.5, before=9, after=4, bold=True)
    set_style(styles['Caption'], 'Calibri', 9, before=5, after=4)
    for name, size in (('Benchmark Cell', 9), ('Figure Note', 8.5), ('APA Reference', 9.5)):
        set_style(styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH), 'Calibri', size, after=3)
    styles['Benchmark Cell'].paragraph_format.space_before = Pt(4)
    styles['APA Reference'].paragraph_format.left_indent = Inches(.18)
    styles['APA Reference'].paragraph_format.first_line_indent = Inches(-.18)
    references = False
    lines = iter(args.source.read_text(encoding='utf-8').splitlines())
    for line in lines:
        if line.startswith('# '):
            document.add_paragraph(line[2:], 'Title')
        elif line.startswith('## '):
            heading = document.add_paragraph(line[3:], 'Heading 1')
            if line.startswith('## 4.4 '):
                heading.paragraph_format.page_break_before = True
            references = line == '## References'
        elif line.startswith('|'):
            table = [line]
            for following in lines:
                if not following.startswith('|'):
                    if following.strip():
                        raise ValueError('The Markdown table must end with a blank line')
                    break
                table.append(following)
            append_table(document, table)
        elif line.startswith('!['):
            alt, path = re.fullmatch(r'!\[([^]]+)]\(([^)]+)\)', line).groups()
            figure = document.add_picture(str(args.source.parent / path), width=Inches(6.67))
            figure._inline.docPr.set('descr', alt)
            document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            document.paragraphs[-1].paragraph_format.space_after = Pt(2)
            document.paragraphs[-1].paragraph_format.keep_with_next = True
        elif line.startswith('**Figure ') or (line.startswith('**') and line.endswith('**')):
            caption = document.add_paragraph(style='Caption')
            caption.add_run(line.replace('**', '')).bold = True
            if not line.startswith('**Figure '):
                caption.paragraph_format.keep_with_next = True
        elif line.startswith('*Note.*'):
            add_inline(document.add_paragraph(style='Figure Note'), line)
        elif line:
            add_inline(document.add_paragraph(style='APA Reference' if references else 'Normal'), line)
    title_ppr = styles['Title']._element.get_or_add_pPr()
    border = title_ppr.find(qn('w:pBdr'))
    if border is not None:
        title_ppr.remove(border)
    document.save(args.output or args.source.with_suffix('.docx'))


if __name__ == '__main__':
    main()
