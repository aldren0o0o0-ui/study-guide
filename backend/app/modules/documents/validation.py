import hashlib
import os
import re
import zipfile
from dataclasses import dataclass
from tempfile import SpooledTemporaryFile

import filetype
from fastapi import UploadFile

from app.core.exceptions import BadRequestException, PayloadTooLargeException
from app.modules.documents.storage import BinaryReadableStream

SUPPORTED_EXTENSIONS: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}

ALLOWED_CLIENT_MIMES: dict[str, set[str]] = {
    "pdf": {"application/pdf", "application/octet-stream", ""},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
        "",
    },
    "txt": {"text/plain", "application/octet-stream", ""},
}

READ_CHUNK_SIZE = 64 * 1024
SPOOL_MAX_MEMORY_BYTES = 2 * 1024 * 1024  # Spill to disk above 2 MB
HEADER_SAMPLE_BYTES = 8192
CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class ValidatedUpload:
    original_filename: str
    file_extension: str
    mime_type: str
    file_size_bytes: int
    checksum_sha256: str
    stream: SpooledTemporaryFile[bytes]

    def close(self) -> None:
        self.stream.close()


def sanitize_filename(raw_filename: str | None) -> tuple[str, str]:
    """Sanitizes a client filename and returns `(safe_display_filename, extension)`."""
    if not raw_filename or not raw_filename.strip():
        raise BadRequestException(
            "Filename is required",
            code="INVALID_FILENAME",
        )

    # Normalize slashes and strip any directory path segments
    normalized = raw_filename.replace("\\", "/")
    base_name = os.path.basename(normalized).strip()
    base_name = CONTROL_CHARS_RE.sub("", base_name).strip()

    if not base_name or base_name in (".", ".."):
        raise BadRequestException(
            "Filename is invalid",
            code="INVALID_FILENAME",
        )

    if "." not in base_name:
        raise BadRequestException(
            "File must have a valid extension (.pdf, .docx, or .txt)",
            code="UNSUPPORTED_FILE_TYPE",
        )

    stem, raw_ext = base_name.rsplit(".", 1)
    stem = stem.strip()
    ext = raw_ext.strip().lower()

    if not stem:
        raise BadRequestException(
            "Filename is invalid",
            code="INVALID_FILENAME",
        )

    if ext not in SUPPORTED_EXTENSIONS:
        raise BadRequestException(
            f"Unsupported file format '.{ext}'. Supported formats are PDF, DOCX, and TXT.",
            code="UNSUPPORTED_FILE_TYPE",
        )

    safe_stem = stem[:200]
    safe_filename = f"{safe_stem}.{ext}"
    return safe_filename, ext


def _validate_client_declared_mime(extension: str, client_content_type: str | None) -> None:
    if not client_content_type:
        return
    # Strip parameters like "; charset=utf-8"
    cleaned_mime = client_content_type.split(";", 1)[0].strip().lower()
    allowed = ALLOWED_CLIENT_MIMES.get(extension, set())
    if cleaned_mime not in allowed:
        raise BadRequestException(
            f"Client Content-Type '{cleaned_mime}' does not match '.{extension}' file format",
            code="MIME_TYPE_MISMATCH",
        )


def _validate_pdf_signature(header: bytes) -> None:
    guessed = filetype.guess(header)
    if guessed is None or guessed.mime != "application/pdf" or not header.startswith(b"%PDF-"):
        raise BadRequestException(
            "File content does not match a valid PDF signature",
            code="INVALID_FILE_SIGNATURE",
        )


def _validate_docx_signature(header: bytes, stream: BinaryReadableStream) -> None:
    guessed = filetype.guess(header)
    if guessed is None or guessed.mime not in (
        "application/zip",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        raise BadRequestException(
            "File content does not match a valid DOCX archive signature",
            code="INVALID_FILE_SIGNATURE",
        )

    stream.seek(0)
    try:
        with zipfile.ZipFile(stream, "r") as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise BadRequestException(
                    "ZIP archive is missing required DOCX structure",
                    code="INVALID_FILE_SIGNATURE",
                )
    except zipfile.BadZipFile as exc:
        raise BadRequestException(
            "Corrupted or invalid DOCX archive",
            code="INVALID_FILE_SIGNATURE",
        ) from exc
    finally:
        stream.seek(0)


def _validate_txt_signature(header: bytes, stream: BinaryReadableStream) -> None:
    guessed = filetype.guess(header)
    if guessed is not None:
        raise BadRequestException(
            "Binary file content is not allowed for TXT documents",
            code="INVALID_FILE_SIGNATURE",
        )

    if b"\x00" in header:
        raise BadRequestException(
            "TXT document contains binary null bytes",
            code="INVALID_FILE_SIGNATURE",
        )

    stream.seek(0)
    decoder_sample = stream.read(64 * 1024)
    stream.seek(0)
    if b"\x00" in decoder_sample:
        raise BadRequestException(
            "TXT document contains binary null bytes",
            code="INVALID_FILE_SIGNATURE",
        )
    try:
        decoder_sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BadRequestException(
            "TXT document must be valid UTF-8 encoded text",
            code="INVALID_FILE_SIGNATURE",
        ) from exc


def validate_and_buffer_upload(
    upload_file: UploadFile,
    *,
    max_size_bytes: int,
) -> ValidatedUpload:
    """Streams an uploaded file, validating size, extension, MIME, and magic-byte signature."""
    safe_filename, extension = sanitize_filename(upload_file.filename)
    _validate_client_declared_mime(extension, upload_file.content_type)

    spooled: SpooledTemporaryFile[bytes] = SpooledTemporaryFile(  # noqa: SIM115
        max_size=SPOOL_MAX_MEMORY_BYTES,
        mode="w+b",
    )

    try:
        hasher = hashlib.sha256()
        total_bytes = 0
        header_buf = bytearray()

        upload_file.file.seek(0)
        while True:
            chunk = upload_file.file.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_size_bytes:
                max_mb = max_size_bytes // (1024 * 1024)
                raise PayloadTooLargeException(
                    f"File exceeds the maximum allowed size of {max_mb} MB",
                    code="FILE_TOO_LARGE",
                )
            if len(header_buf) < HEADER_SAMPLE_BYTES:
                needed = HEADER_SAMPLE_BYTES - len(header_buf)
                header_buf.extend(chunk[:needed])
            hasher.update(chunk)
            spooled.write(chunk)

        if total_bytes == 0:
            raise BadRequestException(
                "Uploaded file is empty",
                code="EMPTY_FILE",
            )

        header = bytes(header_buf)
        if extension == "pdf":
            _validate_pdf_signature(header)
        elif extension == "docx":
            _validate_docx_signature(header, spooled)
        elif extension == "txt":
            _validate_txt_signature(header, spooled)

        spooled.seek(0)
        return ValidatedUpload(
            original_filename=safe_filename,
            file_extension=extension,
            mime_type=SUPPORTED_EXTENSIONS[extension],
            file_size_bytes=total_bytes,
            checksum_sha256=hasher.hexdigest(),
            stream=spooled,
        )
    except Exception:
        spooled.close()
        raise
