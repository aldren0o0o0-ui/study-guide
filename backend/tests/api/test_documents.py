import io
import zipfile
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import ServiceUnavailableException
from app.modules.documents.models import Document
from app.modules.documents.storage import LocalStorageProvider
from tests.unit.test_documents_storage_and_validation import (
    make_valid_docx_bytes,
    make_valid_pdf_bytes,
)


def _register_and_get_headers(client: TestClient, email: str) -> dict[str, str]:
    res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "Password123"},
    )
    assert res.status_code == 201
    token = res.json()["data"]["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_upload_valid_pdf_docx_and_txt_and_download(
    client: TestClient,
    storage_provider: LocalStorageProvider,
    db_session: Session,
) -> None:
    headers = _register_and_get_headers(client, "uploader@example.com")

    # 1. Valid PDF
    pdf_bytes = make_valid_pdf_bytes()
    res_pdf = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("Biology_Chapter_1.pdf", pdf_bytes, "application/pdf")},
    )
    assert res_pdf.status_code == 201
    pdf_doc = res_pdf.json()["data"]
    assert pdf_doc["original_filename"] == "Biology_Chapter_1.pdf"
    assert pdf_doc["file_extension"] == "pdf"
    assert pdf_doc["mime_type"] == "application/pdf"
    assert pdf_doc["file_size_bytes"] == len(pdf_bytes)
    assert pdf_doc["processing_status"] == "queued"
    assert "storage_key" not in pdf_doc

    # Verify stored in private local storage and downloadable
    db_doc = db_session.scalar(select(Document).where(Document.id == pdf_doc["id"]))
    assert db_doc is not None
    assert storage_provider.exists(db_doc.storage_key) is True

    dl_res = client.get(f"/api/v1/documents/{pdf_doc['id']}/download", headers=headers)
    assert dl_res.status_code == 200
    assert dl_res.content == pdf_bytes
    assert dl_res.headers["x-content-type-options"] == "nosniff"

    # 2. Valid DOCX
    docx_bytes = make_valid_docx_bytes()
    res_docx = client.post(
        "/api/v1/documents",
        headers=headers,
        files={
            "file": (
                "Lecture_Notes.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert res_docx.status_code == 201
    assert res_docx.json()["data"]["file_extension"] == "docx"

    # 3. Valid TXT
    txt_bytes = b"Photosynthesis converts light energy into chemical energy.\n"
    res_txt = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("Summary.txt", txt_bytes, "text/plain")},
    )
    assert res_txt.status_code == 201
    assert res_txt.json()["data"]["file_extension"] == "txt"


def test_upload_rejects_invalid_extensions_signatures_mimes_empty_and_oversized(
    client: TestClient,
) -> None:
    headers = _register_and_get_headers(client, "validator@example.com")

    # Unsupported extension
    res_ext = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("image.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert res_ext.status_code == 400
    assert res_ext.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

    # Invalid PDF signature (plain text named .pdf)
    res_bad_pdf = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("fake.pdf", b"Not a real PDF file", "application/pdf")},
    )
    assert res_bad_pdf.status_code == 400
    assert res_bad_pdf.json()["error"]["code"] == "INVALID_FILE_SIGNATURE"

    # Invalid DOCX signature (plain ZIP without word/document.xml)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("hello.txt", "just a zip")
    res_bad_docx = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("fake.docx", zip_buf.getvalue(), "application/zip")},
    )
    assert res_bad_docx.status_code == 400
    assert res_bad_docx.json()["error"]["code"] == "INVALID_FILE_SIGNATURE"

    # Invalid TXT signature (PDF bytes disguised as .txt)
    res_bad_txt = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("disguised.txt", make_valid_pdf_bytes(), "text/plain")},
    )
    assert res_bad_txt.status_code == 400
    assert res_bad_txt.json()["error"]["code"] == "INVALID_FILE_SIGNATURE"

    # MIME mismatch (declaring image/jpeg for .pdf)
    res_mime = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("notes.pdf", make_valid_pdf_bytes(), "image/jpeg")},
    )
    assert res_mime.status_code == 400
    assert res_mime.json()["error"]["code"] == "MIME_TYPE_MISMATCH"

    # Empty file
    res_empty = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert res_empty.status_code == 400
    assert res_empty.json()["error"]["code"] == "EMPTY_FILE"

    # Oversized file
    with patch("app.modules.documents.service.get_settings") as mock_settings:
        cfg = mock_settings.return_value
        cfg.MAX_UPLOAD_SIZE_BYTES = 10  # 10 bytes max for test
        cfg.STORAGE_BACKEND = "local"
        res_large = client.post(
            "/api/v1/documents",
            headers=headers,
            files={"file": ("large.txt", b"0123456789ABCDEF", "text/plain")},
        )
    assert res_large.status_code == 413
    assert res_large.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_duplicate_filenames_and_document_lifecycle(
    client: TestClient,
    storage_provider: LocalStorageProvider,
    db_session: Session,
) -> None:
    headers = _register_and_get_headers(client, "organizer@example.com")

    # Upload two documents with identical filenames
    up1 = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("Week1.txt", b"First version content", "text/plain")},
    )
    up2 = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": ("Week1.txt", b"Second version content", "text/plain")},
    )
    assert up1.status_code == 201
    assert up2.status_code == 201
    doc1_id = up1.json()["data"]["id"]
    doc2_id = up2.json()["data"]["id"]
    assert doc1_id != doc2_id

    # Pagination
    all_docs = client.get("/api/v1/documents?limit=1&offset=0", headers=headers)
    assert all_docs.status_code == 200
    assert all_docs.json()["data"]["total"] == 2
    assert len(all_docs.json()["data"]["items"]) == 1

    # Rename document
    rename_res = client.patch(
        f"/api/v1/documents/{doc2_id}",
        headers=headers,
        json={"original_filename": "Week1_Revised.txt"},
    )
    assert rename_res.status_code == 200
    assert rename_res.json()["data"]["original_filename"] == "Week1_Revised.txt"

    # Delete document -> removes DB row and storage object
    db_doc1 = db_session.scalar(select(Document).where(Document.id == doc1_id))
    assert db_doc1 is not None
    key1 = db_doc1.storage_key
    assert storage_provider.exists(key1) is True

    del_doc = client.delete(f"/api/v1/documents/{doc1_id}", headers=headers)
    assert del_doc.status_code == 200
    assert storage_provider.exists(key1) is False
    assert client.get(f"/api/v1/documents/{doc1_id}", headers=headers).status_code == 404


def test_cross_user_isolation_and_storage_consistency(
    client: TestClient,
    storage_provider: LocalStorageProvider,
) -> None:
    user_a = _register_and_get_headers(client, "alice@example.com")
    user_b = _register_and_get_headers(client, "bob@example.com")

    # Unauthenticated request rejected
    assert client.get("/api/v1/documents").status_code == 401

    # Alice uploads a document
    alice_doc = client.post(
        "/api/v1/documents",
        headers=user_a,
        files={"file": ("alice.txt", b"Alice private study notes", "text/plain")},
    ).json()["data"]

    # Bob cannot view, download, or delete Alice's document
    assert client.get(f"/api/v1/documents/{alice_doc['id']}", headers=user_b).status_code == 404
    assert (
        client.get(f"/api/v1/documents/{alice_doc['id']}/download", headers=user_b).status_code
        == 404
    )
    assert client.delete(f"/api/v1/documents/{alice_doc['id']}", headers=user_b).status_code == 404

    # Database/storage consistency: if DB insert fails after storage.save, stored file is deleted
    files_before = list(storage_provider.base_dir.rglob("*.*"))
    with patch(
        "app.modules.documents.repository.DocumentRepository.create_document",
        side_effect=RuntimeError("Simulated DB failure"),
    ):
        db_fail_res = client.post(
            "/api/v1/documents",
            headers=user_a,
            files={"file": ("rollback.txt", b"Should be cleaned up", "text/plain")},
        )
    assert db_fail_res.status_code == 503
    assert db_fail_res.json()["error"]["code"] == "DOCUMENT_SAVE_FAILED"
    files_after = list(storage_provider.base_dir.rglob("*.*"))
    assert len(files_after) == len(files_before)

    # Storage failure during upload returns 503 cleanly
    with patch.object(
        storage_provider,
        "save",
        side_effect=ServiceUnavailableException(
            "Failed to store uploaded document", code="STORAGE_WRITE_FAILED"
        ),
    ):
        storage_fail_res = client.post(
            "/api/v1/documents",
            headers=user_a,
            files={"file": ("fail.txt", b"Storage down", "text/plain")},
        )
    assert storage_fail_res.status_code == 503
    assert storage_fail_res.json()["error"]["code"] == "STORAGE_WRITE_FAILED"
