import uuid
from collections.abc import Sequence
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from tests.integration.test_processing_and_retrieval import DeterministicKeywordEmbeddingProvider

from app.modules.chat.exceptions import ChatTransientError
from app.modules.chat.generator import GeneratedChatAnswer, assign_chat_source_keys
from app.modules.chat.models import ChatConversation, ChatMessage
from app.modules.chat.router import get_chat_service
from app.modules.chat.schemas import ChatSourceCitation
from app.modules.chat.service import ChatService
from app.modules.documents.models import Document, ProcessingStatus
from app.modules.documents.storage import LocalStorageProvider
from app.modules.processing.chunking import ChunkingService
from app.modules.processing.retrieval import RetrievalService, RetrievedChunk
from app.modules.processing.service import DocumentProcessingService


class DeterministicChatGenerator:
    """Mock chat generator for deterministic integration testing."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        custom_answer: str | None = None,
        has_sufficient_context: bool = True,
    ) -> None:
        self.should_fail = should_fail
        self.custom_answer = custom_answer
        self.has_sufficient_context = has_sufficient_context
        self.recorded_calls: list[dict[str, object]] = []

    def generate_answer(
        self,
        *,
        chunks: Sequence[RetrievedChunk],
        conversation_history: Sequence[dict[str, str]],
        user_query: str,
    ) -> GeneratedChatAnswer:
        self.recorded_calls.append(
            {
                "chunks_count": len(chunks),
                "conversation_history": list(conversation_history),
                "user_query": user_query,
            }
        )
        if self.should_fail:
            raise ChatTransientError(
                "Simulated Gemini rate limit during chat answer generation",
                code="CHAT_RATE_LIMITED",
                user_message=(
                    "AI rate limit reached while generating answer. Please try again in a moment."
                ),
            )

        ordered, _ = assign_chat_source_keys(chunks)
        citations: list[ChatSourceCitation] = []
        for key, chunk in ordered[:2]:
            citations.append(
                ChatSourceCitation(
                    source_key=key,
                    document_id=chunk.document_id,
                    original_filename=chunk.original_filename,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    excerpt=chunk.text[:120],
                )
            )

        if self.custom_answer:
            answer_text = self.custom_answer
        elif citations:
            answer_text = (
                f"According to the source materials [S1], {user_query} is explained in detail."
            )
        else:
            answer_text = (
                "Based on the provided documents, I could not find information about your query."
            )

        return GeneratedChatAnswer(
            answer=answer_text,
            citations=citations,
            has_sufficient_context=self.has_sufficient_context,
        )


def _register_user(client: TestClient, email: str) -> dict[str, str]:
    res = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "Password123"},
    )
    assert res.status_code == 201
    token = res.json()["data"]["tokens"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_ready_doc(
    client: TestClient,
    headers: dict[str, str],
    filename: str,
    content: str,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> uuid.UUID:
    res = client.post(
        "/api/v1/documents",
        headers=headers,
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
    )
    assert res.status_code == 201
    doc_id = uuid.UUID(res.json()["data"]["id"])

    processor = DocumentProcessingService(
        db_session,
        storage=storage_provider,
        embedding_provider=DeterministicKeywordEmbeddingProvider(),
        chunker=ChunkingService(chunk_size=100, chunk_overlap=20),
    )
    processed = processor.process_document(doc_id)
    assert processed is not None
    assert processed.processing_status == ProcessingStatus.READY.value
    return doc_id


def test_chat_conversation_lifecycle_and_multi_turn_qa(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers = _register_user(client, "chat_student@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "mitosis_notes.txt",
        "Mitosis is the process of nuclear division in eukaryotic cells. "
        "During prophase chromatin condenses into chromosomes. "
        "During metaphase chromosomes align at the equatorial plate. "
        "Anaphase separates sister chromatids toward opposite poles.",
        db_session,
        storage_provider,
    )

    mock_generator = DeterministicChatGenerator()

    # Wire mock chat generator through ChatService
    def override_chat_service() -> ChatService:
        retrieval = RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        )
        return ChatService(db_session, retrieval_service=retrieval, generator=mock_generator)

    cast(FastAPI, client.app).dependency_overrides[get_chat_service] = override_chat_service

    # 1. Create conversation with default title
    conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers,
        json={"document_ids": [str(doc_id)]},
    )
    assert conv_res.status_code == 201
    conv_data = conv_res.json()["data"]
    conv_id = uuid.UUID(conv_data["id"])
    assert conv_data["title"] == "Chat: mitosis_notes.txt"
    assert len(conv_data["source_documents"]) == 1
    assert conv_data["source_documents"][0]["document_id"] == str(doc_id)
    assert len(conv_data["messages"]) == 0

    # 2. List conversations
    list_res = client.get("/api/v1/chat/conversations", headers=headers)
    assert list_res.status_code == 200
    list_data = list_res.json()["data"]
    assert list_data["total"] == 1
    assert list_data["items"][0]["id"] == str(conv_id)
    assert list_data["items"][0]["documents_count"] == 1

    # 3. Send first message
    msg1_res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        headers=headers,
        json={"message": "What is mitosis?"},
    )
    assert msg1_res.status_code == 201
    msg1_data = msg1_res.json()["data"]
    assert msg1_data["user_message"]["content"] == "What is mitosis?"
    assert msg1_data["user_message"]["role"] == "user"
    assert msg1_data["assistant_message"]["role"] == "assistant"
    assert "mitosis" in msg1_data["assistant_message"]["content"]
    assert len(msg1_data["assistant_message"]["citations"]) > 0
    assert msg1_data["assistant_message"]["citations"][0]["source_key"] == "S1"
    assert msg1_data["assistant_message"]["citations"][0]["document_id"] == str(doc_id)

    # Verify conversation title auto-updated to query snippet
    assert msg1_data["conversation_title"] == "What is mitosis?"

    # 4. Send follow-up message (multi-turn conversation test)
    msg2_res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        headers=headers,
        json={"message": "What happens during anaphase?"},
    )
    assert msg2_res.status_code == 201

    # Verify generator received previous turn in conversation history
    assert len(mock_generator.recorded_calls) == 2
    second_call_history = mock_generator.recorded_calls[1]["conversation_history"]
    assert isinstance(second_call_history, list)
    assert len(second_call_history) >= 2
    assert second_call_history[0]["role"] == "user"
    assert second_call_history[0]["content"] == "What is mitosis?"
    assert second_call_history[1]["role"] == "assistant"

    # 5. Fetch full conversation detail
    detail_res = client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers)
    assert detail_res.status_code == 200
    detail_data = detail_res.json()["data"]
    assert len(detail_data["messages"]) == 4  # 2 user + 2 assistant messages

    # 6. Fetch conversation messages endpoint
    messages_res = client.get(f"/api/v1/chat/conversations/{conv_id}/messages", headers=headers)
    assert messages_res.status_code == 200
    messages_data = messages_res.json()["data"]
    assert len(messages_data) == 4


def test_chat_multi_document_conversation(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers = _register_user(client, "multi_doc_student@example.com")
    doc1_id = _create_ready_doc(
        client,
        headers,
        "bio_chapter1.txt",
        "Cellular biology focuses on mitochondria, the powerhouse of the cell generating ATP.",
        db_session,
        storage_provider,
    )
    doc2_id = _create_ready_doc(
        client,
        headers,
        "chem_chapter1.txt",
        "Thermochemistry examines enthalpy and entropy changes during chemical reactions.",
        db_session,
        storage_provider,
    )

    mock_generator = DeterministicChatGenerator()

    def override_chat_service() -> ChatService:
        retrieval = RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        )
        return ChatService(db_session, retrieval_service=retrieval, generator=mock_generator)

    cast(FastAPI, client.app).dependency_overrides[get_chat_service] = override_chat_service

    # Create conversation with 2 documents
    conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers,
        json={
            "document_ids": [str(doc1_id), str(doc2_id)],
            "title": "Bio and Chem Combined Study",
        },
    )
    assert conv_res.status_code == 201
    conv_data = conv_res.json()["data"]
    assert conv_data["title"] == "Bio and Chem Combined Study"
    assert len(conv_data["source_documents"]) == 2

    # Ask question
    msg_res = client.post(
        f"/api/v1/chat/conversations/{conv_data['id']}/messages",
        headers=headers,
        json={"message": "Compare ATP and enthalpy."},
    )
    assert msg_res.status_code == 201
    assert len(mock_generator.recorded_calls) == 1
    # Generator should have received chunks from retrieval
    assert int(str(mock_generator.recorded_calls[0]["chunks_count"])) > 0


def test_chat_document_validation_rules(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers_user_a = _register_user(client, "user_a@example.com")
    headers_user_b = _register_user(client, "user_b@example.com")

    doc_a = _create_ready_doc(
        client,
        headers_user_a,
        "doc_a.txt",
        "Some text for doc A",
        db_session,
        storage_provider,
    )

    # 1. Empty document_ids rejected by schema (422)
    empty_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers_user_a,
        json={"document_ids": []},
    )
    assert empty_res.status_code == 422

    # 2. Non-existent document ID rejected (400)
    fake_id = str(uuid.uuid4())
    fake_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers_user_a,
        json={"document_ids": [fake_id]},
    )
    assert fake_res.status_code == 400
    assert fake_res.json()["error"]["code"] == "DOCUMENTS_NOT_FOUND"

    # 3. User B cannot use User A's document (400 DOCUMENTS_NOT_FOUND)
    cross_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers_user_b,
        json={"document_ids": [str(doc_a)]},
    )
    assert cross_res.status_code == 400
    assert cross_res.json()["error"]["code"] == "DOCUMENTS_NOT_FOUND"

    # 4. Unready document rejected (400 DOCUMENTS_NOT_READY)
    doc_res = client.post(
        "/api/v1/documents",
        headers=headers_user_a,
        files={"file": ("unready.txt", b"not processed yet", "text/plain")},
    )
    assert doc_res.status_code == 201
    unready_id = doc_res.json()["data"]["id"]

    unready_conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers_user_a,
        json={"document_ids": [unready_id]},
    )
    assert unready_conv_res.status_code == 400
    assert unready_conv_res.json()["error"]["code"] == "DOCUMENTS_NOT_READY"


def test_chat_cross_user_isolation_and_delete_cascade(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers_user_a = _register_user(client, "isolation_a@example.com")
    headers_user_b = _register_user(client, "isolation_b@example.com")

    doc_id = _create_ready_doc(
        client,
        headers_user_a,
        "isolation_test.txt",
        "Secret study material for user A only.",
        db_session,
        storage_provider,
    )

    mock_generator = DeterministicChatGenerator()

    def override_chat_service() -> ChatService:
        retrieval = RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        )
        return ChatService(db_session, retrieval_service=retrieval, generator=mock_generator)

    cast(FastAPI, client.app).dependency_overrides[get_chat_service] = override_chat_service

    # User A creates conversation
    conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers_user_a,
        json={"document_ids": [str(doc_id)]},
    )
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["data"]["id"]

    # User B cannot view User A's conversation (404)
    b_view_res = client.get(f"/api/v1/chat/conversations/{conv_id}", headers=headers_user_b)
    assert b_view_res.status_code == 404

    # User B cannot message User A's conversation (404)
    b_msg_res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        headers=headers_user_b,
        json={"message": "Let me intrude"},
    )
    assert b_msg_res.status_code == 404

    # User B cannot delete User A's conversation (404)
    b_del_res = client.delete(f"/api/v1/chat/conversations/{conv_id}", headers=headers_user_b)
    assert b_del_res.status_code == 404

    # User A adds a message
    client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        headers=headers_user_a,
        json={"message": "Hello study guide!"},
    )

    # Verify messages are in DB
    msg_count = db_session.execute(
        select(ChatMessage).where(ChatMessage.conversation_id == uuid.UUID(conv_id))
    ).scalars().all()
    assert len(msg_count) == 2

    # User A deletes conversation
    del_res = client.delete(f"/api/v1/chat/conversations/{conv_id}", headers=headers_user_a)
    assert del_res.status_code == 200
    assert del_res.json()["data"]["deleted"] is True

    # Check conversation is deleted from DB
    deleted_conv = db_session.get(ChatConversation, uuid.UUID(conv_id))
    assert deleted_conv is None

    # Check messages are deleted due to cascade
    msgs_after = db_session.execute(
        select(ChatMessage).where(ChatMessage.conversation_id == uuid.UUID(conv_id))
    ).scalars().all()
    assert len(msgs_after) == 0

    # Check document remains intact
    doc_after = db_session.get(Document, doc_id)
    assert doc_after is not None


def test_chat_transient_error_handling(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers = _register_user(client, "rate_limit_student@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "rate_limit_doc.txt",
        "Document content for rate limit test.",
        db_session,
        storage_provider,
    )

    failing_generator = DeterministicChatGenerator(should_fail=True)

    def override_chat_service() -> ChatService:
        retrieval = RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        )
        return ChatService(db_session, retrieval_service=retrieval, generator=failing_generator)

    cast(FastAPI, client.app).dependency_overrides[get_chat_service] = override_chat_service

    conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers,
        json={"document_ids": [str(doc_id)]},
    )
    conv_id = conv_res.json()["data"]["id"]

    msg_res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        headers=headers,
        json={"message": "Will trigger rate limit"},
    )
    # Transient error should return 503
    assert msg_res.status_code == 503
    assert msg_res.json()["error"]["code"] == "CHAT_RATE_LIMITED"
    assert "rate limit" in msg_res.json()["error"]["message"].lower()

    # Verify no assistant or user message was committed on failure (clean transaction boundary)
    messages = db_session.execute(
        select(ChatMessage).where(ChatMessage.conversation_id == uuid.UUID(conv_id))
    ).scalars().all()
    assert len(messages) == 0


def test_chat_conversation_history_window_exactly_six_prior_messages(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers = _register_user(client, "history_window_student@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "history_study.txt",
        "Deep learning is a subset of machine learning using multi-layer neural networks.",
        db_session,
        storage_provider,
    )

    mock_generator = DeterministicChatGenerator()

    def override_chat_service() -> ChatService:
        retrieval = RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        )
        return ChatService(db_session, retrieval_service=retrieval, generator=mock_generator)

    cast(FastAPI, client.app).dependency_overrides[get_chat_service] = override_chat_service

    conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers,
        json={"document_ids": [str(doc_id)]},
    )
    conv_id = conv_res.json()["data"]["id"]

    # Send 4 messages (4 turns = 8 messages in conversation)
    for i in range(1, 5):
        res = client.post(
            f"/api/v1/chat/conversations/{conv_id}/messages",
            headers=headers,
            json={"message": f"Question {i}"},
        )
        assert res.status_code == 201

    # Send 5th message (active query). Prior messages in DB = 8. Window limit = 6.
    res5 = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        headers=headers,
        json={"message": "Question 5"},
    )
    assert res5.status_code == 201

    # Inspect the 5th call to mock_generator
    call_5 = mock_generator.recorded_calls[4]
    history = call_5["conversation_history"]
    assert isinstance(history, list)
    # Must be capped at exactly 6 messages (the last 3 turns)
    assert len(history) == 6
    # Active query must NOT be in the history
    assert call_5["user_query"] == "Question 5"
    history_contents = [h["content"] for h in history]
    assert "Question 5" not in history_contents
    # Oldest messages (Question 1) must have rolled out
    assert "Question 1" not in history_contents
    # Chronological ordering: Question 2 turn -> Question 3 turn -> Question 4 turn
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Question 2"
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "Question 3"
    assert history[4]["role"] == "user"
    assert history[4]["content"] == "Question 4"


def test_chat_message_deterministic_ordering_with_equal_timestamps(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers = _register_user(client, "timestamp_order_student@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "ordering_notes.txt",
        "Deterministic ordering verification document.",
        db_session,
        storage_provider,
    )

    conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers,
        json={"document_ids": [str(doc_id)]},
    )
    conv_id = uuid.UUID(conv_res.json()["data"]["id"])

    # Directly insert 3 messages with the exact same timestamp
    from datetime import UTC, datetime
    fixed_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Deterministic UUIDs to test tie-breaking
    id_low = uuid.UUID("10000000-0000-0000-0000-000000000001")
    id_mid = uuid.UUID("20000000-0000-0000-0000-000000000002")
    id_high = uuid.UUID("30000000-0000-0000-0000-000000000003")

    msg_high = ChatMessage(
        id=id_high,
        conversation_id=conv_id,
        role="assistant",
        content="Third message",
        created_at=fixed_time,
    )
    msg_low = ChatMessage(
        id=id_low,
        conversation_id=conv_id,
        role="user",
        content="First message",
        created_at=fixed_time,
    )
    msg_mid = ChatMessage(
        id=id_mid,
        conversation_id=conv_id,
        role="assistant",
        content="Second message",
        created_at=fixed_time,
    )

    db_session.add_all([msg_high, msg_low, msg_mid])
    db_session.commit()

    # Query via messages endpoint
    res = client.get(f"/api/v1/chat/conversations/{conv_id}/messages", headers=headers)
    assert res.status_code == 200
    msgs = res.json()["data"]
    assert len(msgs) == 3
    # Ordered by (created_at.asc(), id.asc())
    assert msgs[0]["id"] == str(id_low)
    assert msgs[1]["id"] == str(id_mid)
    assert msgs[2]["id"] == str(id_high)


def test_chat_create_conversation_with_failing_initial_message_rolls_back(
    client: TestClient,
    db_session: Session,
    storage_provider: LocalStorageProvider,
) -> None:
    headers = _register_user(client, "atomic_create_student@example.com")
    doc_id = _create_ready_doc(
        client,
        headers,
        "atomic_study.txt",
        "Verification of atomic conversation creation and message turn.",
        db_session,
        storage_provider,
    )

    failing_generator = DeterministicChatGenerator(should_fail=True)

    def override_chat_service() -> ChatService:
        retrieval = RetrievalService(
            db_session,
            embedding_provider=DeterministicKeywordEmbeddingProvider(),
        )
        return ChatService(db_session, retrieval_service=retrieval, generator=failing_generator)

    cast(FastAPI, client.app).dependency_overrides[get_chat_service] = override_chat_service

    # Creation with initial_message when generator fails
    conv_res = client.post(
        "/api/v1/chat/conversations",
        headers=headers,
        json={
            "document_ids": [str(doc_id)],
            "initial_message": "This will fail in generator",
        },
    )
    assert conv_res.status_code == 503
    assert conv_res.json()["error"]["code"] == "CHAT_RATE_LIMITED"

    # Verify conversation was rolled back and does NOT exist in DB
    list_res = client.get("/api/v1/chat/conversations", headers=headers)
    assert list_res.status_code == 200
    assert list_res.json()["data"]["total"] == 0
