import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.modules.analysis.models import AnalysisDocument, AnalysisStatus, DocumentAnalysis


class AnalysisRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_analysis_for_user(
        self,
        analysis_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> DocumentAnalysis | None:
        stmt = (
            select(DocumentAnalysis)
            .options(
                selectinload(DocumentAnalysis.analysis_documents).selectinload(
                    AnalysisDocument.document
                )
            )
            .where(
                DocumentAnalysis.id == analysis_id,
                DocumentAnalysis.user_id == user_id,
            )
        )
        return self.db.scalar(stmt)

    def list_analyses_for_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[DocumentAnalysis], int]:
        count_stmt = select(func.count(DocumentAnalysis.id)).where(
            DocumentAnalysis.user_id == user_id
        )
        total = int(self.db.scalar(count_stmt) or 0)

        items_stmt = (
            select(DocumentAnalysis)
            .options(
                selectinload(DocumentAnalysis.analysis_documents).selectinload(
                    AnalysisDocument.document
                )
            )
            .where(DocumentAnalysis.user_id == user_id)
            .order_by(DocumentAnalysis.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list(self.db.scalars(items_stmt).all())
        return items, total

    def create_analysis(
        self,
        *,
        user_id: uuid.UUID,
        title: str,
        focus_topic: str | None,
        document_ids: Sequence[uuid.UUID],
    ) -> DocumentAnalysis:
        analysis = DocumentAnalysis(
            user_id=user_id,
            title=title[:255],
            focus_topic=focus_topic[:300] if focus_topic else None,
            status=AnalysisStatus.QUEUED.value,
        )
        self.db.add(analysis)
        self.db.flush()

        for pos, doc_id in enumerate(document_ids):
            link = AnalysisDocument(
                analysis_id=analysis.id,
                document_id=doc_id,
                position=pos,
            )
            self.db.add(link)

        self.db.flush()
        return analysis

    def delete_analysis(self, analysis: DocumentAnalysis) -> None:
        self.db.delete(analysis)
        self.db.flush()
