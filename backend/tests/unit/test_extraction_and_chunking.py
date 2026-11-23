import io

import docx
import pymupdf
import pytest

from app.modules.processing.chunking import ChunkingService
from app.modules.processing.exceptions import DocumentExtractionError
from app.modules.processing.extraction import (
    ExtractedSegment,
    extract_document_content,
    normalize_extracted_text,
)


def make_multipage_pdf_bytes(pages: list[str]) -> bytes:
    pdf = pymupdf.open()
    try:
        for text in pages:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        return bytes(pdf.tobytes())
    finally:
        pdf.close()


def make_scanned_image_only_pdf_bytes() -> bytes:
    pdf = pymupdf.open()
    try:
        page = pdf.new_page()
        # Create a 10x10 RGB pixmap and insert it as an image with zero text
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.set_rect(pix.irect, (200, 200, 200))
        img_bytes = pix.tobytes("png")
        page.insert_image(pymupdf.Rect(50, 50, 150, 150), stream=img_bytes)
        return bytes(pdf.tobytes())
    finally:
        pdf.close()


def make_structured_docx_bytes() -> bytes:
    doc = docx.Document()
    doc.add_heading("Chapter 1: Cellular Respiration", level=1)
    doc.add_paragraph("Glycolysis occurs in the cytoplasm of the cell.")
    doc.add_paragraph("The Krebs cycle takes place inside the mitochondria.")

    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Stage"
    table.cell(0, 1).text = "ATP Yield"
    table.cell(1, 0).text = "Glycolysis"
    table.cell(1, 1).text = "2 Net ATP"

    doc.add_heading("Chapter 2: Electron Transport Chain", level=2)
    doc.add_paragraph("Oxidative phosphorylation produces the majority of cellular ATP.")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_pdf_preserves_page_numbers_and_normalizes_whitespace() -> None:
    pdf_bytes = make_multipage_pdf_bytes(
        [
            "Page 1: Introduction to Thermodynamics.\n\n\n\nFirst Law: Energy conservation.",
            "",  # Blank intermediate page should be skipped cleanly
            "Page 3: Second Law and Entropy   with   extra   spaces.",
        ]
    )
    extracted = extract_document_content(pdf_bytes, file_extension="pdf")
    assert extracted.file_extension == "pdf"
    assert extracted.page_count == 3
    assert len(extracted.segments) == 2

    assert extracted.segments[0].page_number == 1
    assert "Introduction to Thermodynamics." in extracted.segments[0].text
    assert "\n\n\n" not in extracted.segments[0].text

    assert extracted.segments[1].page_number == 3
    assert "Second Law and Entropy with extra spaces." in extracted.segments[1].text


def test_extract_docx_preserves_headings_paragraphs_and_tables() -> None:
    docx_bytes = make_structured_docx_bytes()
    extracted = extract_document_content(docx_bytes, file_extension="docx")

    assert extracted.file_extension == "docx"
    assert extracted.page_count is None
    assert len(extracted.segments) == 2

    seg1 = extracted.segments[0]
    assert seg1.page_number is None
    assert seg1.section_title == "Chapter 1: Cellular Respiration"
    assert "Glycolysis occurs in the cytoplasm" in seg1.text
    assert "Stage | ATP Yield" in seg1.text
    assert "Glycolysis | 2 Net ATP" in seg1.text

    seg2 = extracted.segments[1]
    assert seg2.page_number is None
    assert seg2.section_title == "Chapter 2: Electron Transport Chain"
    assert "Oxidative phosphorylation" in seg2.text


def test_extract_txt_decodes_utf8_and_normalizes_whitespace() -> None:
    raw = (
        "\ufeffLine one   with\t\ttabs.\r\n\r\n\r\n\r\nLine two with UTF-8: ΔG = ΔH - TΔS.\r\n"
    ).encode()
    extracted = extract_document_content([raw[:15], raw[15:]], file_extension="txt")

    assert extracted.file_extension == "txt"
    assert extracted.page_count is None
    assert len(extracted.segments) == 1
    assert extracted.segments[0].page_number is None
    assert (
        extracted.segments[0].text == "Line one with tabs.\n\nLine two with UTF-8: ΔG = ΔH - TΔS."
    )


def test_extract_handles_empty_scanned_corrupted_oversized_and_unsupported_files() -> None:
    # 1. Empty byte payload
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(b"", file_extension="txt")
    assert exc_info.value.code == "EMPTY_DOCUMENT_CONTENT"

    # 2. Whitespace-only TXT
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(b"   \r\n\t  \n ", file_extension="txt")
    assert exc_info.value.code == "EMPTY_DOCUMENT_CONTENT"

    # 3. Scanned image-only PDF without OCR text layer
    scanned_pdf = make_scanned_image_only_pdf_bytes()
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(scanned_pdf, file_extension="pdf")
    assert exc_info.value.code == "OCR_NOT_SUPPORTED"

    # 4. Blank PDF with no text or images
    blank_pdf = make_multipage_pdf_bytes([""])
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(blank_pdf, file_extension="pdf")
    assert exc_info.value.code == "EMPTY_DOCUMENT_CONTENT"

    # 5. Corrupted PDF
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(b"%PDF-1.4 corrupted payload", file_extension="pdf")
    assert exc_info.value.code == "CORRUPTED_DOCUMENT"

    # 6. Corrupted DOCX
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(b"PK\x03\x04 not a valid zip", file_extension="docx")
    assert exc_info.value.code == "CORRUPTED_DOCUMENT"

    # 7. Invalid UTF-8 or null byte TXT
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(b"\xff\xfe invalid utf8", file_extension="txt")
    assert exc_info.value.code == "CORRUPTED_DOCUMENT"

    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(b"null\x00byte", file_extension="txt")
    assert exc_info.value.code == "CORRUPTED_DOCUMENT"

    # 8. Unsupported extension
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(b"hello", file_extension="pptx")
    assert exc_info.value.code == "UNSUPPORTED_DOCUMENT_FORMAT"

    # 9. Oversized stream
    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document_content(
            [b"0123456789", b"ABCDEF"], file_extension="txt", max_size_bytes=12
        )
    assert exc_info.value.code == "DOCUMENT_TOO_LARGE"


def test_chunking_short_normal_long_overlap_deterministic_and_page_metadata() -> None:
    chunker = ChunkingService(chunk_size=120, chunk_overlap=60)

    # 1. Short document fits in a single chunk without duplicate overlap chunks
    short_segments = [
        ExtractedSegment(
            text="Mitosis is the process of nuclear division in eukaryotic cells.",
            page_number=1,
        )
    ]
    short_chunks = chunker.chunk_segments(short_segments)
    assert len(short_chunks) == 1
    assert short_chunks[0].chunk_index == 0
    assert short_chunks[0].page_start == 1
    assert short_chunks[0].page_end == 1

    # 2. Multi-page document with overlap and cross-page boundary preservation
    multi_segments = [
        ExtractedSegment(
            text=(
                "Sentence one on page one introduces cell theory. "
                "Sentence two on page one explains membrane transport."
            ),
            page_number=1,
        ),
        ExtractedSegment(
            text=(
                "Sentence three on page two covers active transport pumps. "
                "Sentence four on page two describes endocytosis and exocytosis in detail."
            ),
            page_number=2,
        ),
    ]
    chunks_run_1 = chunker.chunk_segments(multi_segments)
    chunks_run_2 = chunker.chunk_segments(multi_segments)

    # Deterministic ordering and hashes
    assert len(chunks_run_1) >= 2
    assert [c.content_hash for c in chunks_run_1] == [c.content_hash for c in chunks_run_2]
    assert [c.chunk_index for c in chunks_run_1] == list(range(len(chunks_run_1)))

    # Verify overlap carries trailing sentence from chunk 0 into chunk 1
    assert "Sentence two on page one" in chunks_run_1[0].content
    assert "Sentence two on page one" in chunks_run_1[1].content
    # Chunk 1 combines overlap from page 1 with new text from page 2 -> page_start=1, page_end=2
    assert chunks_run_1[1].page_start == 1
    assert chunks_run_1[1].page_end == 2

    # 3. Single oversized sentence without punctuation splits cleanly without exceeding chunk_size
    long_sentence = " ".join([f"word{i}" for i in range(60)])
    long_chunks = chunker.chunk_segments(
        [ExtractedSegment(text=long_sentence, page_number=None, section_title="Glossary")]
    )
    assert len(long_chunks) >= 3
    for chunk in long_chunks:
        assert 0 < chunk.char_count <= 120
        assert chunk.page_start is None
        assert chunk.page_end is None
        assert chunk.section_title == "Glossary"

    # 4. Empty text raises DocumentExtractionError
    with pytest.raises(DocumentExtractionError) as exc_info:
        chunker.chunk_segments([ExtractedSegment(text="   \n\n  ")])
    assert exc_info.value.code == "EMPTY_DOCUMENT_CONTENT"
    assert normalize_extracted_text("") == ""
