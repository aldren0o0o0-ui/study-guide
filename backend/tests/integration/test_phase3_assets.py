import uuid

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database.vector import DEFAULT_EMBEDDING_DIMENSION
from app.modules.documents.storage import StorageProvider
from app.modules.processing.service import DocumentProcessingService


class DeterministicKeywordEmbeddingProvider:
    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.01] * self.dimension


def _register_and_get_headers(client: TestClient, email: str) -> dict[str, str]:
    res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "Password123"},
    )
    assert res.status_code == 201
    token = res.json()["data"]["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_test_pdf_with_image() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 50), "Figure 1: Photosynthesis in Chloroplasts")

    # Insert a 60x60 test image
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.set_rect(pix.irect, (50, 180, 80))
    img_bytes = pix.tobytes("png")
    page.insert_image(pymupdf.Rect(50, 100, 200, 250), stream=img_bytes)

    pdf_bytes = bytes(pdf.tobytes())
    pdf.close()
    return pdf_bytes


def test_document_asset_lifecycle_and_security(
    client: TestClient,
    db_session: Session,
    storage_provider: StorageProvider,
) -> None:
    u1_suffix = uuid.uuid4().hex[:8]
    u2_suffix = uuid.uuid4().hex[:8]
    auth_headers = _register_and_get_headers(client, f"user1_{u1_suffix}@example.com")
    second_user_headers = _register_and_get_headers(client, f"user2_{u2_suffix}@example.com")

    # 1. Upload PDF with an embedded image
    pdf_bytes = _create_test_pdf_with_image()
    upload_res = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("biology_lab.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_data = upload_res.json()["data"]
    doc_id = doc_data["id"]

    # 2. Process document synchronously via DocumentProcessingService with mock embedding
    processing_service = DocumentProcessingService(
        db_session,
        storage=storage_provider,
        embedding_provider=DeterministicKeywordEmbeddingProvider(),
    )
    processed_doc = processing_service.process_document(uuid.UUID(doc_id))
    assert processed_doc is not None
    assert processed_doc.processing_status == "ready"

    # 3. List assets via GET /api/v1/documents/{document_id}/assets
    assets_res = client.get(
        f"/api/v1/documents/{doc_id}/assets",
        headers=auth_headers,
    )
    assert assets_res.status_code == 200
    assets_data = assets_res.json()["data"]
    assert len(assets_data) == 1
    asset = assets_data[0]
    assert asset["document_id"] == doc_id
    assert asset["page_number"] == 1
    assert asset["mime_type"] == "image/png"
    asset_id = asset["id"]

    # 4. Stream asset via GET /api/v1/documents/{document_id}/assets/{asset_id}
    download_res = client.get(
        f"/api/v1/documents/{doc_id}/assets/{asset_id}",
        headers=auth_headers,
    )
    assert download_res.status_code == 200
    assert download_res.headers["content-type"] == "image/png"
    assert len(download_res.content) > 0

    # 5. Security: Second user cannot access User 1's assets
    second_user_list_res = client.get(
        f"/api/v1/documents/{doc_id}/assets",
        headers=second_user_headers,
    )
    assert second_user_list_res.status_code == 404

    second_user_asset_res = client.get(
        f"/api/v1/documents/{doc_id}/assets/{asset_id}",
        headers=second_user_headers,
    )
    assert second_user_asset_res.status_code == 404

    # 6. Delete document and verify cascading cleanup
    del_res = client.delete(
        f"/api/v1/documents/{doc_id}",
        headers=auth_headers,
    )
    assert del_res.status_code == 200

    # Verify asset is deleted from database
    asset_after_del = client.get(
        f"/api/v1/documents/{doc_id}/assets",
        headers=auth_headers,
    )
    assert asset_after_del.status_code == 404
