import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.modules.processing.exceptions import DocumentExtractionError
from app.modules.processing.extraction import (
    ExtractedDocument,
    ExtractedSegment,
    normalize_extracted_text,
)

# Sentence boundary regex: splits after ., !, ? followed by whitespace, keeping punctuation
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    """A deterministic chunk of document text with source attribution metadata."""

    chunk_index: int
    content: str
    content_hash: str
    char_count: int
    page_start: int | None
    page_end: int | None
    section_title: str | None


@dataclass(frozen=True)
class _AtomicUnit:
    text: str
    separator_before: str
    page_number: int | None
    section_title: str | None


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _split_long_sentence(
    sentence: str,
    *,
    max_len: int,
    page_number: int | None,
    section_title: str | None,
) -> list[_AtomicUnit]:
    """Splits a single sentence that exceeds `max_len` along word boundaries, then hard slices."""
    words = sentence.split(" ")
    units: list[_AtomicUnit] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        if not word:
            continue
        if len(word) > max_len:
            if current_words:
                joined = " ".join(current_words)
                units.append(
                    _AtomicUnit(
                        text=joined,
                        separator_before=" " if units else "",
                        page_number=page_number,
                        section_title=section_title,
                    )
                )
                current_words = []
                current_len = 0
            for start in range(0, len(word), max_len):
                piece = word[start : start + max_len]
                units.append(
                    _AtomicUnit(
                        text=piece,
                        separator_before=" " if units else "",
                        page_number=page_number,
                        section_title=section_title,
                    )
                )
            continue

        added_len = len(word) if not current_words else len(word) + 1
        if current_len + added_len <= max_len:
            current_words.append(word)
            current_len += added_len
        else:
            joined = " ".join(current_words)
            units.append(
                _AtomicUnit(
                    text=joined,
                    separator_before=" " if units else "",
                    page_number=page_number,
                    section_title=section_title,
                )
            )
            current_words = [word]
            current_len = len(word)

    if current_words:
        joined = " ".join(current_words)
        units.append(
            _AtomicUnit(
                text=joined,
                separator_before=" " if units else "",
                page_number=page_number,
                section_title=section_title,
            )
        )

    return units


def _decompose_segments(
    segments: Sequence[ExtractedSegment],
    *,
    chunk_size: int,
) -> list[_AtomicUnit]:
    """Breaks extracted segments into paragraph and sentence units smaller than `chunk_size`."""
    units: list[_AtomicUnit] = []

    for seg in segments:
        normalized = normalize_extracted_text(seg.text)
        if not normalized:
            continue

        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        for para in paragraphs:
            # Preserve Markdown tables and display equations as atomic structural units
            is_table = para.startswith("|") and ("\n|" in para or para.endswith("|"))
            is_display_math = para.startswith("$$") and para.endswith("$$")

            if (is_table or is_display_math) and len(para) <= chunk_size:
                units.append(
                    _AtomicUnit(
                        text=para,
                        separator_before="\n\n",
                        page_number=seg.page_number,
                        section_title=seg.section_title,
                    )
                )
                continue

            # If a table exceeds chunk_size, split by row groups while repeating headers
            if is_table and len(para) > chunk_size:
                table_lines = [
                    line_item.strip() for line_item in para.split("\n") if line_item.strip()
                ]
                if len(table_lines) >= 3 and table_lines[1].startswith("|"):
                    header_block = f"{table_lines[0]}\n{table_lines[1]}"
                    current_rows: list[str] = []
                    current_len = len(header_block)

                    for data_row in table_lines[2:]:
                        row_len = len(data_row) + 1
                        if current_len + row_len <= chunk_size or not current_rows:
                            current_rows.append(data_row)
                            current_len += row_len
                        else:
                            slice_text = f"{header_block}\n" + "\n".join(current_rows)
                            units.append(
                                _AtomicUnit(
                                    text=slice_text,
                                    separator_before="\n\n",
                                    page_number=seg.page_number,
                                    section_title=seg.section_title,
                                    )
                            )
                            current_rows = [data_row]
                            current_len = len(header_block) + row_len

                    if current_rows:
                        slice_text = f"{header_block}\n" + "\n".join(current_rows)
                        units.append(
                            _AtomicUnit(
                                text=slice_text,
                                separator_before="\n\n",
                                page_number=seg.page_number,
                                section_title=seg.section_title,
                            )
                        )
                    continue

            # Standard paragraph: split into lines and sentences
            sentences: list[str] = []
            for line in para.split("\n"):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(stripped_line) if s.strip()]
                sentences.extend(parts)

            first_in_para = True
            for sentence in sentences:
                sep = "\n\n" if first_in_para else " "
                first_in_para = False
                if len(sentence) <= chunk_size:
                    units.append(
                        _AtomicUnit(
                            text=sentence,
                            separator_before=sep,
                            page_number=seg.page_number,
                            section_title=seg.section_title,
                        )
                    )
                else:
                    sub_units = _split_long_sentence(
                        sentence,
                        max_len=chunk_size,
                        page_number=seg.page_number,
                        section_title=seg.section_title,
                    )
                    for idx, sub in enumerate(sub_units):
                        units.append(
                            _AtomicUnit(
                                text=sub.text,
                                separator_before=sep if idx == 0 else " ",
                                page_number=sub.page_number,
                                section_title=sub.section_title,
                            )
                        )

    return units


def _join_units(units: Sequence[_AtomicUnit]) -> str:
    if not units:
        return ""
    parts: list[str] = [units[0].text]
    for unit in units[1:]:
        parts.append(unit.separator_before)
        parts.append(unit.text)
    return "".join(parts).strip()


def _build_chunk_from_units(chunk_index: int, units: Sequence[_AtomicUnit]) -> TextChunk:
    content = _join_units(units)
    pages = [u.page_number for u in units if u.page_number is not None]
    page_start = min(pages) if pages else None
    page_end = max(pages) if pages else None

    section_title: str | None = None
    for u in units:
        if u.section_title:
            section_title = u.section_title
            break

    return TextChunk(
        chunk_index=chunk_index,
        content=content,
        content_hash=compute_content_hash(content),
        char_count=len(content),
        page_start=page_start,
        page_end=page_end,
        section_title=section_title,
    )


class ChunkingService:
    """
    Deterministic, sentence- and paragraph-aware document chunker.

    Independent of any embedding provider or database layer.
    """

    def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size < 50:
            raise ValueError("chunk_size must be at least 50 characters")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, extracted: ExtractedDocument) -> list[TextChunk]:
        return self.chunk_segments(extracted.segments)

    def chunk_segments(self, segments: Sequence[ExtractedSegment]) -> list[TextChunk]:
        units = _decompose_segments(segments, chunk_size=self.chunk_size)
        if not units:
            raise DocumentExtractionError(
                "No non-empty text chunks could be created from document",
                code="EMPTY_DOCUMENT_CONTENT",
                user_message="This document does not contain any readable text to process.",
            )

        chunks: list[TextChunk] = []
        current_units: list[_AtomicUnit] = []
        current_len = 0

        for unit in units:
            sep_len = len(unit.separator_before) if current_units else 0
            candidate_len = current_len + sep_len + len(unit.text)

            if current_units and candidate_len > self.chunk_size:
                chunk = _build_chunk_from_units(len(chunks), current_units)
                if chunk.content and (not chunks or chunk.content_hash != chunks[-1].content_hash):
                    chunks.append(chunk)

                # Compute overlap tail from `current_units` that also fits with the incoming `unit`
                overlap_units: list[_AtomicUnit] = []
                overlap_len = 0
                if self.chunk_overlap > 0:
                    for prev_unit in reversed(current_units):
                        added = (
                            len(prev_unit.text)
                            if not overlap_units
                            else len(prev_unit.text) + len(overlap_units[-1].separator_before)
                        )
                        if overlap_len + added > self.chunk_overlap:
                            break
                        if (
                            overlap_len + added + len(unit.separator_before) + len(unit.text)
                            > self.chunk_size
                        ):
                            break
                        overlap_units.append(prev_unit)
                        overlap_len += added
                    overlap_units.reverse()

                current_units = list(overlap_units)
                current_units.append(unit)
                current_len = len(_join_units(current_units))
            else:
                current_units.append(unit)
                current_len = candidate_len

        if current_units:
            final_chunk = _build_chunk_from_units(len(chunks), current_units)
            if final_chunk.content and (
                not chunks or final_chunk.content_hash != chunks[-1].content_hash
            ):
                chunks.append(final_chunk)

        return chunks
