import io
import uuid
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from app.core.exceptions import BadRequestException, ServiceUnavailableException
from app.modules.documents.storage import (
    LocalStorageProvider,
    S3StorageProvider,
    build_storage_key,
    validate_storage_key,
)
from app.modules.documents.validation import sanitize_filename


def make_valid_pdf_bytes() -> bytes:
    return b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n"


def make_valid_docx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")
        zf.writestr("word/document.xml", "<w:document><w:body><w:p/></w:body></w:document>")
    return buf.getvalue()


def test_sanitize_filename_strips_paths_and_rejects_unsupported() -> None:
    safe_name, ext = sanitize_filename("../../secret/My Study Notes.PDF")
    assert safe_name == "My Study Notes.pdf"
    assert ext == "pdf"

    with pytest.raises(BadRequestException) as exc_info:
        sanitize_filename("malware.exe")
    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"

    with pytest.raises(BadRequestException) as exc_info_empty:
        sanitize_filename("   ")
    assert exc_info_empty.value.code == "INVALID_FILENAME"


def test_storage_key_prevents_path_traversal() -> None:
    uid = uuid.uuid4()
    did = uuid.uuid4()
    valid_key = build_storage_key(uid, did, "pdf")
    validate_storage_key(valid_key)

    for bad_key in [
        "../outside.pdf",
        f"users/{uid}/../../outside.pdf",
        f"users/{uid}/documents/{did}.exe",
        "/etc/passwd",
    ]:
        with pytest.raises(BadRequestException) as exc_info:
            validate_storage_key(bad_key)
        assert exc_info.value.code == "INVALID_STORAGE_KEY"


def test_local_storage_provider_save_read_delete(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path / "local_store")
    key = build_storage_key(uuid.uuid4(), uuid.uuid4(), "txt")
    content = b"StudyGuide local storage test content"

    assert provider.exists(key) is False
    provider.save(key, io.BytesIO(content), "text/plain")
    assert provider.exists(key) is True

    read_back = b"".join(provider.open_stream(key))
    assert read_back == content

    # Duplicate key write rejected
    with pytest.raises(ServiceUnavailableException) as exc_collision:
        provider.save(key, io.BytesIO(content), "text/plain")
    assert exc_collision.value.code == "STORAGE_COLLISION"

    provider.delete(key)
    assert provider.exists(key) is False


def test_s3_storage_provider_operations_and_error_handling() -> None:
    mock_s3 = MagicMock()
    provider = S3StorageProvider(bucket_name="test-bucket", s3_client=mock_s3)
    key = build_storage_key(uuid.uuid4(), uuid.uuid4(), "pdf")

    provider.save(key, io.BytesIO(make_valid_pdf_bytes()), "application/pdf")
    assert mock_s3.upload_fileobj.called

    assert provider.exists(key) is True
    provider.delete(key)
    mock_s3.delete_object.assert_called_once_with(Bucket="test-bucket", Key=key)

    # Simulate S3 404 on exists()
    mock_s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}},
        "HeadObject",
    )
    assert provider.exists(key) is False

    # Simulate S3 write failure
    mock_s3.upload_fileobj.side_effect = ClientError(
        {"Error": {"Code": "InternalError", "Message": "S3 down"}},
        "PutObject",
    )
    with pytest.raises(ServiceUnavailableException) as exc_info:
        provider.save(key, io.BytesIO(make_valid_pdf_bytes()), "application/pdf")
    assert exc_info.value.code == "STORAGE_WRITE_FAILED"
