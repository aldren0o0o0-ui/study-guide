import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.modules.analysis.queue import get_analysis_queue
from app.modules.analysis.schemas import (
    AnalysisCreateRequest,
    AnalysisDeleteResponse,
    AnalysisDetailRead,
    AnalysisListResponse,
)
from app.modules.analysis.service import AnalysisQueueDispatcher, AnalysisService
from app.modules.auth.dependencies import CurrentUser
from app.modules.exams.router import get_exam_service
from app.modules.exams.schemas import ExamCreateRequest, ExamReviewerDetailRead
from app.modules.exams.service import ExamReviewerService
from app.shared.responses import ApiResponse

analyses_router = APIRouter(prefix="/analyses", tags=["Analyses"])


def get_analysis_service(
    db: Annotated[Session, Depends(get_db)],
    analysis_queue: Annotated[AnalysisQueueDispatcher, Depends(get_analysis_queue)],
) -> AnalysisService:
    return AnalysisService(db, analysis_queue=analysis_queue)


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]


@analyses_router.post(
    "",
    response_model=ApiResponse[AnalysisDetailRead],
    status_code=status.HTTP_201_CREATED,
)
def create_analysis(
    payload: AnalysisCreateRequest,
    current_user: CurrentUser,
    service: AnalysisServiceDep,
) -> ApiResponse[AnalysisDetailRead]:
    analysis = service.create_analysis(current_user.id, payload)
    return ApiResponse(data=analysis, message="Analysis generation queued")


@analyses_router.get(
    "",
    response_model=ApiResponse[AnalysisListResponse],
    status_code=status.HTTP_200_OK,
)
def list_analyses(
    current_user: CurrentUser,
    service: AnalysisServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[AnalysisListResponse]:
    result = service.list_analyses(current_user.id, limit=limit, offset=offset)
    return ApiResponse(data=result)


@analyses_router.get(
    "/{analysis_id}",
    response_model=ApiResponse[AnalysisDetailRead],
    status_code=status.HTTP_200_OK,
)
def get_analysis(
    analysis_id: uuid.UUID,
    current_user: CurrentUser,
    service: AnalysisServiceDep,
) -> ApiResponse[AnalysisDetailRead]:
    analysis = service.get_analysis(current_user.id, analysis_id)
    return ApiResponse(data=analysis)


@analyses_router.post(
    "/{analysis_id}/retry",
    response_model=ApiResponse[AnalysisDetailRead],
    status_code=status.HTTP_200_OK,
)
def retry_analysis(
    analysis_id: uuid.UUID,
    current_user: CurrentUser,
    service: AnalysisServiceDep,
) -> ApiResponse[AnalysisDetailRead]:
    analysis = service.retry_analysis(current_user.id, analysis_id)
    return ApiResponse(data=analysis, message="Analysis retry queued")


@analyses_router.delete(
    "/{analysis_id}",
    response_model=ApiResponse[AnalysisDeleteResponse],
    status_code=status.HTTP_200_OK,
)
def delete_analysis(
    analysis_id: uuid.UUID,
    current_user: CurrentUser,
    service: AnalysisServiceDep,
) -> ApiResponse[AnalysisDeleteResponse]:
    result = service.delete_analysis(current_user.id, analysis_id)
    return ApiResponse(data=result, message="Analysis deleted")


@analyses_router.post(
    "/{analysis_id}/quiz",
    response_model=ApiResponse[ExamReviewerDetailRead],
    status_code=status.HTTP_201_CREATED,
)
def create_quiz_from_analysis(
    analysis_id: uuid.UUID,
    current_user: CurrentUser,
    exam_service: Annotated[ExamReviewerService, Depends(get_exam_service)],
    payload: ExamCreateRequest | None = None,
) -> ApiResponse[ExamReviewerDetailRead]:
    req = payload or ExamCreateRequest(analysis_id=analysis_id)
    if req.analysis_id is None:
        req.analysis_id = analysis_id
    reviewer = exam_service.create_reviewer(current_user.id, req)
    return ApiResponse(data=reviewer, message="Practice quiz generation queued")
