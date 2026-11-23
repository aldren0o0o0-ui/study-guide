import io
import uuid
import xml.etree.ElementTree as ET

import docx
import pymupdf
import pytest

from app.core.exceptions import BadRequestException
from app.modules.documents.storage import (
    build_asset_storage_key,
    validate_storage_key,
)
from app.modules.processing.chunking import ChunkingService
from app.modules.processing.extraction import (
    ExtractedSegment,
    _format_markdown_table,
    extract_document_content,
)
from app.modules.processing.omml import convert_omml_to_latex_math


def test_markdown_table_formatter() -> None:
    raw_rows = [
        ["Header 1", "Header 2", "Header 3"],
        ["Cell A\nwith newline", "Cell B | with pipe", "Cell C"],
        ["Cell D", "Cell E", "Cell F"],
    ]
    md = _format_markdown_table(raw_rows)
    assert "| Header 1 | Header 2 | Header 3 |" in md
    assert "| :--- | :--- | :--- |" in md
    assert "Cell A <br> with newline" in md
    assert r"Cell B \| with pipe" in md
    assert "| Cell D | Cell E | Cell F |" in md


def test_omml_to_latex_fraction_and_exponents() -> None:
    # <m:f><m:num><m:r><m:t>1</m:t></m:r></m:num><m:den><m:r><m:t>2</m:t></m:r></m:den></m:f>
    xml = """<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:f>
            <m:num><m:r><m:t>1</m:t></m:r></m:num>
            <m:den><m:r><m:t>2</m:t></m:r></m:den>
        </m:f>
    </m:oMath>"""
    elem = ET.fromstring(xml)
    latex = convert_omml_to_latex_math(elem)
    assert latex == r"$\frac{1}{2}$"

    # Pythagoras: a^2 + b^2 = c^2
    pythagoras_xml = """<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:sSup><m:e><m:r><m:t>a</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
        <m:r><m:t> + </m:t></m:r>
        <m:sSup><m:e><m:r><m:t>b</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
        <m:r><m:t> = </m:t></m:r>
        <m:sSup><m:e><m:r><m:t>c</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
    </m:oMath>"""
    pyth_elem = ET.fromstring(pythagoras_xml)
    pyth_latex = convert_omml_to_latex_math(pyth_elem)
    assert pyth_latex == r"${a}^{2} + {b}^{2} = {c}^{2}$"


def test_omml_to_latex_subscript_and_radical() -> None:
    xml = """<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:sSub><m:e><m:r><m:t>H</m:t></m:r></m:e><m:sub><m:r><m:t>2</m:t></m:r></m:sub></m:sSub>
        <m:r><m:t>O</m:t></m:r>
    </m:oMath>"""
    elem = ET.fromstring(xml)
    assert convert_omml_to_latex_math(elem) == r"${H}_{2}O$"

    rad_xml = """<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:rad>
            <m:deg><m:r><m:t>3</m:t></m:r></m:deg>
            <m:e><m:r><m:t>x</m:t></m:r></m:e>
        </m:rad>
    </m:oMath>"""
    rad_elem = ET.fromstring(rad_xml)
    assert convert_omml_to_latex_math(rad_elem) == r"$\sqrt[3]{x}$"


def test_extract_docx_with_omml_math() -> None:
    doc = docx.Document()
    doc.add_heading("Algebra and Physics", level=1)
    p = doc.add_paragraph("Einstein proposed ")

    # Manually append an OMML equation element to the paragraph
    omml_xml = """<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:r><m:t>E = m</m:t></m:r>
        <m:sSup><m:e><m:r><m:t>c</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
    </m:oMath>"""
    p._p.append(docx.oxml.parse_xml(omml_xml))

    buf = io.BytesIO()
    doc.save(buf)
    extracted = extract_document_content(buf.getvalue(), file_extension="docx")

    assert len(extracted.segments) == 1
    assert "Einstein proposed" in extracted.segments[0].text
    assert r"$E = m{c}^{2}$" in extracted.segments[0].text


def test_extract_docx_tables_into_markdown() -> None:
    doc = docx.Document()
    doc.add_heading("Periodic Table Excerpt", level=1)
    table = doc.add_table(rows=3, cols=3)
    data = [
        ["Element", "Symbol", "Atomic Number"],
        ["Hydrogen", "H", "1"],
        ["Helium", "He", "2"],
    ]
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            table.cell(r_idx, c_idx).text = val

    buf = io.BytesIO()
    doc.save(buf)
    extracted = extract_document_content(buf.getvalue(), file_extension="docx")

    assert "| Element | Symbol | Atomic Number |" in extracted.segments[0].text
    assert "| :--- | :--- | :--- |" in extracted.segments[0].text
    assert "| Hydrogen | H | 1 |" in extracted.segments[0].text
    assert "| Helium | He | 2 |" in extracted.segments[0].text


def test_chunking_preserves_table_integrity() -> None:
    chunker = ChunkingService(chunk_size=500, chunk_overlap=50)
    table_text = (
        "| Subject | Units | Grade |\n"
        "| :--- | :--- | :--- |\n"
        "| Science | 3 | 1.25 |\n"
        "| Mathematics | 4 | 1.00 |\n"
        "| History | 3 | 1.50 |"
    )
    segment = ExtractedSegment(
        text=f"Below are the semester grades.\n\n{table_text}\n\nAll requirements completed.",
        page_number=1,
    )
    from app.modules.processing.extraction import ExtractedDocument

    doc = ExtractedDocument(
        file_extension="docx",
        segments=[segment],
        page_count=1,
        total_chars=len(segment.text),
    )
    chunks = chunker.chunk_document(doc)
    # The table should be preserved in chunk content with headers intact
    table_chunks = [c for c in chunks if "| Subject | Units | Grade |" in c.content]
    assert len(table_chunks) >= 1
    assert "| Mathematics | 4 | 1.00 |" in table_chunks[0].content


def test_asset_storage_key_generation_and_validation() -> None:
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    asset_id = uuid.uuid4()

    key = build_asset_storage_key(user_id, doc_id, asset_id, "png")
    assert key == f"users/{user_id}/documents/{doc_id}/assets/{asset_id}.png"
    validate_storage_key(key)

    # Invalid characters or unsupported extensions fail validation
    with pytest.raises(BadRequestException):
        validate_storage_key(f"users/{user_id}/documents/{doc_id}/assets/{asset_id}.exe")

    with pytest.raises(BadRequestException):
        validate_storage_key("users/../../etc/passwd")


def test_pdf_image_extraction() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Figure 1: Laboratory Setup")

    # Insert a 60x60 test image (exceeding 50x50 threshold)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.set_rect(pix.irect, (100, 150, 200))
    img_bytes = pix.tobytes("png")
    page.insert_image(pymupdf.Rect(50, 100, 200, 250), stream=img_bytes)

    pdf_bytes = bytes(pdf.tobytes())
    pdf.close()

    extracted = extract_document_content(pdf_bytes, file_extension="pdf")
    assert extracted.file_extension == "pdf"
    assert "Figure 1: Laboratory Setup" in extracted.segments[0].text
    assert len(extracted.assets) == 1
    assert extracted.assets[0].page_number == 1
    assert extracted.assets[0].mime_type == "image/png"
    assert extracted.assets[0].file_extension == "png"
    assert len(extracted.assets[0].data) > 0
