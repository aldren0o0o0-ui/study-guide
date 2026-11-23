import hashlib
import io
import logging
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field

import docx
import pymupdf
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.modules.processing.exceptions import DocumentExtractionError
from app.modules.processing.omml import convert_omml_to_latex_math

logger = logging.getLogger("app.modules.processing.extraction")

DEFAULT_MAX_EXTRACT_BYTES = 26_214_400  # 25 MB

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

_OMML_MATH_TAGS = {
    "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath",
    "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara",
}


@dataclass(frozen=True)
class ExtractedAsset:
    """An image, diagram, or visual asset extracted from the document."""

    data: bytes
    mime_type: str
    file_extension: str
    page_number: int | None = None
    section_title: str | None = None
    caption: str | None = None
    order_index: int = 0
    width: int | None = None
    height: int | None = None
    checksum_sha256: str = ""


@dataclass(frozen=True)
class ExtractedSegment:
    """A contiguous segment of extracted text with optional page and section metadata."""

    text: str
    page_number: int | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized extraction output for an uploaded document."""

    file_extension: str
    segments: list[ExtractedSegment]
    page_count: int | None
    total_chars: int
    assets: list[ExtractedAsset] = field(default_factory=list)


def normalize_extracted_text(raw_text: str) -> str:
    """
    Normalizes extracted text while preserving paragraph, table, and equation structure.

    - Removes null bytes and non-printable control characters.
    - Converts CRLF/CR line endings to LF.
    - Collapses runs of horizontal spaces/tabs into a single space (preserving table pipes).
    - Trims trailing whitespace on each line.
    - Collapses 3 or more consecutive newlines into a single blank line (`\\n\\n`).
    """
    if not raw_text:
        return ""

    cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)

    lines: list[str] = []
    for line in cleaned.split("\n"):
        collapsed_line = _HORIZONTAL_SPACE_RE.sub(" ", line).strip()
        lines.append(collapsed_line)

    joined = "\n".join(lines)
    return _EXCESS_NEWLINES_RE.sub("\n\n", joined).strip()


def _extension_to_mime(ext: str) -> str:
    clean_ext = ext.lower().lstrip(".")
    if clean_ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if clean_ext == "png":
        return "image/png"
    if clean_ext == "webp":
        return "image/webp"
    if clean_ext == "gif":
        return "image/gif"
    return "application/octet-stream"


def _format_markdown_table(rows: list[list[str]]) -> str:
    """
    Converts a grid of string cells into standard GitHub-Flavored Markdown table syntax.
    Escapes internal pipe characters and converts multiline cell text to maintain row boundaries.
    """
    if not rows:
        return ""

    clean_rows: list[list[str]] = []
    for row in rows:
        cleaned_cells = [
            re.sub(r"[\r\n]+", " <br> ", str(cell or "").strip()).replace("|", r"\|")
            for cell in row
        ]
        if any(c for c in cleaned_cells):
            clean_rows.append(cleaned_cells)

    if not clean_rows:
        return ""

    num_cols = max(len(r) for r in clean_rows)
    if num_cols == 0:
        return ""

    padded: list[list[str]] = []
    for r in clean_rows:
        padded.append(r + [""] * (num_cols - len(r)))

    headers = padded[0]
    if not any(headers):
        headers = [f"Col {i + 1}" for i in range(num_cols)]

    header_line = "| " + " | ".join(headers) + " |"
    delimiter_line = "| " + " | ".join([":---"] * num_cols) + " |"

    body_lines: list[str] = []
    data_rows = padded[1:] if len(padded) > 1 else []
    for r in data_rows:
        body_lines.append("| " + " | ".join(r) + " |")

    return "\n" + "\n".join([header_line, delimiter_line] + body_lines) + "\n"


def _read_bounded_bytes(
    source: bytes | Iterable[bytes],
    *,
    max_size_bytes: int,
) -> bytes:
    if isinstance(source, bytes):
        if len(source) == 0:
            raise DocumentExtractionError(
                "Document stream is empty",
                code="EMPTY_DOCUMENT_CONTENT",
                user_message="The uploaded document is empty and contains no text to process.",
            )
        if len(source) > max_size_bytes:
            raise DocumentExtractionError(
                f"Document size {len(source)} exceeds extraction limit {max_size_bytes}",
                code="DOCUMENT_TOO_LARGE",
                user_message="The document exceeds the maximum size allowed for text extraction.",
            )
        return source

    buffer = bytearray()
    for chunk in source:
        if not chunk:
            continue
        buffer.extend(chunk)
        if len(buffer) > max_size_bytes:
            raise DocumentExtractionError(
                f"Document stream exceeded extraction limit {max_size_bytes}",
                code="DOCUMENT_TOO_LARGE",
                user_message="The document exceeds the maximum size allowed for text extraction.",
            )

    if len(buffer) == 0:
        raise DocumentExtractionError(
            "Document stream is empty",
            code="EMPTY_DOCUMENT_CONTENT",
            user_message="The uploaded document is empty and contains no text to process.",
        )
    return bytes(buffer)


def _extract_pdf(raw_bytes: bytes) -> ExtractedDocument:
    try:
        pdf_doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentExtractionError(
            f"Failed to open PDF document: {type(exc).__name__}",
            code="CORRUPTED_DOCUMENT",
            user_message="This PDF file appears to be corrupted or unreadable.",
        ) from exc

    try:
        if getattr(pdf_doc, "is_encrypted", False) or getattr(pdf_doc, "needs_pass", False):
            raise DocumentExtractionError(
                "Encrypted or password-protected PDF cannot be extracted",
                code="ENCRYPTED_DOCUMENT",
                user_message=(
                    "Password-protected PDF files are not supported. Please upload an unlocked PDF."
                ),
            )

        page_count = int(len(pdf_doc))
        if page_count == 0:
            raise DocumentExtractionError(
                "PDF document has zero pages",
                code="EMPTY_DOCUMENT_CONTENT",
                user_message="This PDF document contains no pages.",
            )

        segments: list[ExtractedSegment] = []
        assets: list[ExtractedAsset] = []
        seen_image_hashes: set[str] = set()
        has_images = False

        text_flags = (
            pymupdf.TEXT_PRESERVE_WHITESPACE
            | pymupdf.TEXT_PRESERVE_LIGATURES
            | pymupdf.TEXT_DEHYPHENATE
        )

        for page_index in range(page_count):
            page = pdf_doc[page_index]
            page_num = page_index + 1

            # 1. Extract structured tables via PyMuPDF find_tables()
            table_rects: list[pymupdf.Rect] = []
            page_elements: list[tuple[float, str]] = []  # (y0, content)

            try:
                table_finder = page.find_tables()
                for table in getattr(table_finder, "tables", []):
                    table_data = table.extract()
                    if table_data:
                        md_table = _format_markdown_table(table_data)
                        if md_table:
                            bbox = table.bbox
                            table_rects.append(pymupdf.Rect(bbox))
                            page_elements.append((bbox[1], md_table))
            except Exception as exc:
                logger.warning(
                    "Table extraction error on page %d: %s",
                    page_num,
                    exc,
                )

            # 2. Extract text blocks, skipping blocks inside extracted tables
            try:
                blocks = page.get_text("blocks", flags=text_flags, sort=True)
                for block in blocks:
                    # block: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) >= 7 and block[6] == 0:  # Text block
                        b_rect = pymupdf.Rect(block[:4])
                        # Check if this block falls inside an extracted table
                        inside_table = False
                        b_mid_x = (b_rect.x0 + b_rect.x1) / 2.0
                        b_mid_y = (b_rect.y0 + b_rect.y1) / 2.0
                        for t_rect in table_rects:
                            if (
                                t_rect.x0 <= b_mid_x <= t_rect.x1
                                and t_rect.y0 <= b_mid_y <= t_rect.y1
                            ):
                                inside_table = True
                                break

                        if not inside_table:
                            block_text = str(block[4] or "").strip()
                            if block_text:
                                page_elements.append((b_rect.y0, block_text))
            except Exception:
                # Fallback to standard text extraction if blocks fail
                raw_page_text = str(page.get_text("text", flags=text_flags, sort=True) or "")
                if raw_page_text:
                    page_elements.append((0.0, raw_page_text))

            # Sort page elements by vertical reading order (y0)
            page_elements.sort(key=lambda elem: elem[0])
            combined_page_text = "\n\n".join(elem[1] for elem in page_elements)
            normalized_page_text = normalize_extracted_text(combined_page_text)

            if normalized_page_text:
                segments.append(
                    ExtractedSegment(
                        text=normalized_page_text,
                        page_number=page_num,
                        section_title=None,
                    )
                )

            # 3. Extract embedded images & diagrams
            try:
                images_info = page.get_images(full=True)
                if images_info:
                    has_images = True
                    for img_meta in images_info:
                        xref = img_meta[0]
                        try:
                            base_img = pdf_doc.extract_image(xref)
                            if not base_img:
                                continue
                            img_bytes = base_img.get("image")
                            if not img_bytes:
                                continue

                            width = base_img.get("width")
                            height = base_img.get("height")
                            # Ignore tiny icons / decorative bullets (< 20x20 or < 100 bytes)
                            if (
                                (width and width < 20)
                                or (height and height < 20)
                                or len(img_bytes) < 100
                            ):
                                continue

                            sha256 = hashlib.sha256(img_bytes).hexdigest()
                            if sha256 in seen_image_hashes:
                                continue
                            seen_image_hashes.add(sha256)

                            img_ext = str(base_img.get("ext", "png")).lower()
                            mime = _extension_to_mime(img_ext)

                            assets.append(
                                ExtractedAsset(
                                    data=img_bytes,
                                    mime_type=mime,
                                    file_extension=img_ext,
                                    page_number=page_num,
                                    section_title=None,
                                    caption=f"Figure from page {page_num}",
                                    order_index=len(assets),
                                    width=width,
                                    height=height,
                                    checksum_sha256=sha256,
                                )
                            )
                        except Exception as img_exc:
                            logger.debug("Skipping unreadable PDF image xref %d: %s", xref, img_exc)
            except Exception as exc:
                logger.debug("Failed image extraction on page %d: %s", page_num, exc)

        if not segments:
            if has_images:
                raise DocumentExtractionError(
                    "PDF contains images but no extractable text layer",
                    code="OCR_NOT_SUPPORTED",
                    user_message=(
                        "This PDF appears to contain scanned images without selectable text. "
                        "OCR is not supported in this version."
                    ),
                )
            raise DocumentExtractionError(
                "PDF contains no extractable text across any page",
                code="EMPTY_DOCUMENT_CONTENT",
                user_message="This PDF document does not contain any readable text.",
            )

        total_chars = sum(len(seg.text) for seg in segments)
        return ExtractedDocument(
            file_extension="pdf",
            segments=segments,
            page_count=page_count,
            total_chars=total_chars,
            assets=assets,
        )
    finally:
        pdf_doc.close()


def _is_heading_style(style_name: str | None) -> bool:
    if not style_name:
        return False
    lowered = style_name.strip().lower()
    return lowered.startswith("heading") or lowered in ("title", "subtitle")


def _extract_docx_paragraph_content(
    para_element: CT_P,
    doc: docx.Document,
) -> tuple[str, list[tuple[str, bytes]]]:
    """
    Extracts text, inline OMML math, and embedded image references from a DOCX paragraph element.
    Maintains the sequential order between text runs and mathematical equations.
    """
    parts: list[str] = []
    extracted_images: list[tuple[str, bytes]] = []

    for child in para_element.iterchildren():
        tag = getattr(child, "tag", "")

        # 1. OMML Math equation: convert to LaTeX
        if tag in _OMML_MATH_TAGS:
            latex_math = convert_omml_to_latex_math(
                child,
                is_block=(
                    tag == "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara"
                ),
            )
            if latex_math:
                parts.append(latex_math)
            continue

        # 2. Standard text run <w:r>
        if tag.endswith("}r") or tag == "r":
            # Check for embedded drawing inside the run
            drawings = child.xpath(".//a:blip/@r:embed")
            for rid in drawings:
                try:
                    rel = doc.part.related_parts.get(rid)
                    if rel and hasattr(rel, "blob") and rel.blob:
                        extracted_images.append((str(rel.content_type), rel.blob))
                except Exception:
                    pass

            # Extract run text
            run_texts = child.xpath(".//w:t/text()")
            if run_texts:
                parts.append("".join(run_texts))
            continue

        # 3. Direct drawing element <w:drawing>
        if tag.endswith("}drawing") or tag == "drawing":
            drawings = child.xpath(".//a:blip/@r:embed")
            for rid in drawings:
                try:
                    rel = doc.part.related_parts.get(rid)
                    if rel and hasattr(rel, "blob") and rel.blob:
                        extracted_images.append((str(rel.content_type), rel.blob))
                except Exception:
                    pass

    return "".join(parts).strip(), extracted_images


def _extract_docx(raw_bytes: bytes) -> ExtractedDocument:
    try:
        document = docx.Document(io.BytesIO(raw_bytes))
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise DocumentExtractionError(
            f"Failed to parse DOCX archive: {type(exc).__name__}",
            code="CORRUPTED_DOCUMENT",
            user_message="This DOCX file appears to be corrupted or unreadable.",
        ) from exc
    except Exception as exc:
        raise DocumentExtractionError(
            f"Unexpected DOCX parsing failure: {type(exc).__name__}",
            code="CORRUPTED_DOCUMENT",
            user_message="This DOCX file appears to be corrupted or unreadable.",
        ) from exc

    segments: list[ExtractedSegment] = []
    assets: list[ExtractedAsset] = []
    seen_image_hashes: set[str] = set()
    current_section: str | None = None
    current_paragraphs: list[str] = []

    def _flush_section() -> None:
        if not current_paragraphs:
            return
        combined = normalize_extracted_text("\n\n".join(current_paragraphs))
        if combined:
            segments.append(
                ExtractedSegment(
                    text=combined,
                    page_number=None,
                    section_title=current_section,
                )
            )
        current_paragraphs.clear()

    # Walk body elements in document order (paragraphs and tables)
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            para = Paragraph(child, document)
            style_name = para.style.name if para.style is not None else None

            # Extract text with OMML math and drawing images
            para_text, para_images = _extract_docx_paragraph_content(child, document)

            # Process any images found in paragraph
            for content_type, img_blob in para_images:
                if len(img_blob) < 100:
                    continue
                sha256 = hashlib.sha256(img_blob).hexdigest()
                if sha256 in seen_image_hashes:
                    continue
                seen_image_hashes.add(sha256)

                ext = "png"
                if "jpeg" in content_type:
                    ext = "jpg"
                elif "webp" in content_type:
                    ext = "webp"
                elif "gif" in content_type:
                    ext = "gif"

                assets.append(
                    ExtractedAsset(
                        data=img_blob,
                        mime_type=content_type or _extension_to_mime(ext),
                        file_extension=ext,
                        page_number=None,
                        section_title=current_section,
                        caption=f"Figure in {current_section}" if current_section else "Figure",
                        order_index=len(assets),
                        checksum_sha256=sha256,
                    )
                )

            cleaned_text = normalize_extracted_text(para_text)
            if not cleaned_text:
                continue

            if _is_heading_style(style_name):
                _flush_section()
                current_section = cleaned_text[:255]
                current_paragraphs.append(cleaned_text)
            else:
                current_paragraphs.append(cleaned_text)

        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            raw_rows: list[list[str]] = []
            for row in table.rows:
                row_cells: list[str] = []
                # Deduplicate identical merged cell instances
                seen_tc_ids: set[int] = set()
                for cell in row.cells:
                    tc_id = id(cell._tc)
                    if tc_id in seen_tc_ids:
                        continue
                    seen_tc_ids.add(tc_id)
                    row_cells.append(normalize_extracted_text(cell.text))
                if any(row_cells):
                    raw_rows.append(row_cells)

            if raw_rows:
                md_table = _format_markdown_table(raw_rows)
                if md_table:
                    current_paragraphs.append(md_table.strip())

    _flush_section()

    if not segments:
        raise DocumentExtractionError(
            "DOCX document contains no extractable text",
            code="EMPTY_DOCUMENT_CONTENT",
            user_message="This DOCX document does not contain any readable text.",
        )

    total_chars = sum(len(seg.text) for seg in segments)
    return ExtractedDocument(
        file_extension="docx",
        segments=segments,
        page_count=None,
        total_chars=total_chars,
        assets=assets,
    )


def _extract_txt(raw_bytes: bytes) -> ExtractedDocument:
    if b"\x00" in raw_bytes:
        raise DocumentExtractionError(
            "TXT file contains binary null bytes",
            code="CORRUPTED_DOCUMENT",
            user_message=(
                "This TXT file contains invalid binary data and cannot be read as UTF-8 text."
            ),
        )

    try:
        decoded = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError(
            "TXT file is not valid UTF-8",
            code="CORRUPTED_DOCUMENT",
            user_message="This TXT file is not valid UTF-8 encoded text.",
        ) from exc

    # Strip UTF-8 BOM if present
    decoded = decoded.lstrip("\ufeff")
    normalized = normalize_extracted_text(decoded)
    if not normalized:
        raise DocumentExtractionError(
            "TXT document contains only whitespace",
            code="EMPTY_DOCUMENT_CONTENT",
            user_message="This TXT document contains only whitespace and has no text to process.",
        )

    segment = ExtractedSegment(
        text=normalized,
        page_number=None,
        section_title=None,
    )
    return ExtractedDocument(
        file_extension="txt",
        segments=[segment],
        page_count=None,
        total_chars=len(normalized),
        assets=[],
    )


def extract_document_content(
    source: bytes | Iterable[bytes],
    *,
    file_extension: str,
    max_size_bytes: int = DEFAULT_MAX_EXTRACT_BYTES,
) -> ExtractedDocument:
    """
    Extracts and normalizes text, tables, equations, and visual assets from a PDF, DOCX, or TXT
    byte stream.

    Preserves page numbers for PDF documents, heading/paragraph structure for DOCX,
    structured tables in Markdown, equations in LaTeX, and images as ExtractedAsset items.
    """
    ext = file_extension.strip().lower().lstrip(".")
    if ext not in ("pdf", "docx", "txt"):
        raise DocumentExtractionError(
            f"Unsupported document extension for extraction: '{ext}'",
            code="UNSUPPORTED_DOCUMENT_FORMAT",
            user_message=(
                f"Unsupported document format '.{ext}'. Only PDF, DOCX, and TXT are supported."
            ),
        )

    raw_bytes = _read_bounded_bytes(source, max_size_bytes=max_size_bytes)

    if ext == "pdf":
        return _extract_pdf(raw_bytes)
    if ext == "docx":
        return _extract_docx(raw_bytes)
    return _extract_txt(raw_bytes)
