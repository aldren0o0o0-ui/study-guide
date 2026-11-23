import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.database.session import get_session_factory
from app.main import create_app
from app.modules.analysis.models import AnalysisDocument, DocumentAnalysis
from app.modules.analysis.queue import get_analysis_queue
from app.modules.auth.models import RefreshSession, User
from app.modules.chat.models import (
    ChatConversation,
    ChatConversationDocument,
    ChatMessage,
)
from app.modules.documents.models import Document, DocumentChunk
from app.modules.documents.storage import LocalStorageProvider, get_storage_provider
from app.modules.exams.models import (
    ExamAttempt,
    ExamQuestion,
    ExamReviewer,
    ExamReviewerDocument,
    StudentAnswer,
)
from app.modules.exams.queue import get_exam_queue
from app.modules.processing.queue import get_processing_queue
from app.modules.voice_practice.models import (
    VoicePracticeFeedback,
    VoicePracticeSession,
    VoicePracticeSessionChunk,
    VoicePracticeSessionDocument,
    VoicePracticeTurn,
)


class RecordingProcessingQueue:
    """In-memory test double for `ProcessingQueue` that records enqueued document UUIDs."""

    def __init__(self) -> None:
        self.enqueued_document_ids: list[uuid.UUID] = []

    def enqueue_document(self, document_id: uuid.UUID) -> str:
        self.enqueued_document_ids.append(document_id)
        return f"test-job-{document_id}"


class RecordingAnalysisQueue:
    """In-memory test double for `AnalysisQueue` that records enqueued analysis UUIDs."""

    def __init__(self) -> None:
        self.enqueued_analysis_ids: list[uuid.UUID] = []

    def enqueue_analysis(self, analysis_id: uuid.UUID) -> str:
        self.enqueued_analysis_ids.append(analysis_id)
        return f"test-analysis-job-{analysis_id}"


class RecordingExamQueue:
    """In-memory test double for `ExamQueue` that records enqueued exam reviewer UUIDs."""

    def __init__(self) -> None:
        self.enqueued_exam_ids: list[uuid.UUID] = []

    def enqueue_exam(self, reviewer_id: uuid.UUID) -> str:
        self.enqueued_exam_ids.append(reviewer_id)
        return f"test-exam-job-{reviewer_id}"


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        session.execute(delete(VoicePracticeFeedback))
        session.execute(delete(VoicePracticeTurn))
        session.execute(delete(VoicePracticeSessionChunk))
        session.execute(delete(VoicePracticeSessionDocument))
        session.execute(delete(VoicePracticeSession))
        session.execute(delete(StudentAnswer))
        session.execute(delete(ExamAttempt))
        session.execute(delete(ExamQuestion))
        session.execute(delete(ExamReviewerDocument))
        session.execute(delete(ExamReviewer))
        session.execute(delete(AnalysisDocument))
        session.execute(delete(DocumentAnalysis))
        session.execute(delete(DocumentChunk))
        session.execute(delete(Document))
        session.execute(delete(RefreshSession))
        session.execute(delete(User))
        session.commit()
        yield session
    finally:
        session.rollback()
        session.execute(delete(VoicePracticeFeedback))
        session.execute(delete(VoicePracticeTurn))
        session.execute(delete(VoicePracticeSessionChunk))
        session.execute(delete(VoicePracticeSessionDocument))
        session.execute(delete(VoicePracticeSession))
        session.execute(delete(ChatMessage))
        session.execute(delete(ChatConversationDocument))
        session.execute(delete(ChatConversation))
        session.execute(delete(StudentAnswer))
        session.execute(delete(ExamAttempt))
        session.execute(delete(ExamQuestion))
        session.execute(delete(ExamReviewerDocument))
        session.execute(delete(ExamReviewer))
        session.execute(delete(AnalysisDocument))
        session.execute(delete(DocumentAnalysis))
        session.execute(delete(DocumentChunk))
        session.execute(delete(Document))
        session.execute(delete(RefreshSession))
        session.execute(delete(User))
        session.commit()
        session.close()


@pytest.fixture
def storage_provider(tmp_path: Path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "storage")


@pytest.fixture
def processing_queue() -> RecordingProcessingQueue:
    return RecordingProcessingQueue()


@pytest.fixture
def analysis_queue() -> RecordingAnalysisQueue:
    return RecordingAnalysisQueue()


@pytest.fixture
def exam_queue() -> RecordingExamQueue:
    return RecordingExamQueue()


@pytest.fixture
def client(
    db_session: Session,
    storage_provider: LocalStorageProvider,
    processing_queue: RecordingProcessingQueue,
    analysis_queue: RecordingAnalysisQueue,
    exam_queue: RecordingExamQueue,
) -> Generator[TestClient, None, None]:
    _ = db_session
    app = create_app()
    app.dependency_overrides[get_storage_provider] = lambda: storage_provider
    app.dependency_overrides[get_processing_queue] = lambda: processing_queue
    app.dependency_overrides[get_analysis_queue] = lambda: analysis_queue
    app.dependency_overrides[get_exam_queue] = lambda: exam_queue
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
