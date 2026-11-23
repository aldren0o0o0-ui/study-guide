import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.exams.queue import get_exam_queue
from app.modules.exams.schemas import (
    ExamAttemptStartResponse,
    ExamAttemptSubmitRequest,
    ExamCreateRequest,
    ExamDeleteResponse,
    ExamResultRead,
    ExamReviewerDetailRead,
    ExamReviewerListResponse,
)
from app.modules.exams.service import ExamQueueDispatcher, ExamReviewerService
from app.shared.responses import ApiResponse

exams_router = APIRouter(prefix="/exams", tags=["Exams"])


def get_exam_service(
    db: Annotated[Session, Depends(get_db)],
    exam_queue: Annotated[ExamQueueDispatcher, Depends(get_exam_queue)],
) -> ExamReviewerService:
    return ExamReviewerService(db, exam_queue=exam_queue)


ExamServiceDep = Annotated[ExamReviewerService, Depends(get_exam_service)]


@exams_router.post(
    "",
    response_model=ApiResponse[ExamReviewerDetailRead],
    status_code=status.HTTP_201_CREATED,
)
def create_exam_reviewer(
    payload: ExamCreateRequest,
    current_user: CurrentUser,
    service: ExamServiceDep,
) -> ApiResponse[ExamReviewerDetailRead]:
    reviewer = service.create_reviewer(current_user.id, payload)
    return ApiResponse(data=reviewer, message="Exam reviewer generation queued")


@exams_router.get(
    "",
    response_model=ApiResponse[ExamReviewerListResponse],
    status_code=status.HTTP_200_OK,
)
def list_exam_reviewers(
    current_user: CurrentUser,
    service: ExamServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ExamReviewerListResponse]:
    result = service.list_reviewers(current_user.id, limit=limit, offset=offset)
    return ApiResponse(data=result)


@exams_router.get(
    "/{reviewer_id}",
    response_model=ApiResponse[ExamReviewerDetailRead],
    status_code=status.HTTP_200_OK,
)
def get_exam_reviewer(
    reviewer_id: uuid.UUID,
    current_user: CurrentUser,
    service: ExamServiceDep,
) -> ApiResponse[ExamReviewerDetailRead]:
    reviewer = service.get_reviewer(current_user.id, reviewer_id)
    return ApiResponse(data=reviewer)


@exams_router.post(
    "/{reviewer_id}/retry",
    response_model=ApiResponse[ExamReviewerDetailRead],
    status_code=status.HTTP_200_OK,
)
def retry_exam_reviewer(
    reviewer_id: uuid.UUID,
    current_user: CurrentUser,
    service: ExamServiceDep,
) -> ApiResponse[ExamReviewerDetailRead]:
    reviewer = service.retry_reviewer(current_user.id, reviewer_id)
    return ApiResponse(data=reviewer, message="Exam generation retry queued")


@exams_router.delete(
    "/{reviewer_id}",
    response_model=ApiResponse[ExamDeleteResponse],
    status_code=status.HTTP_200_OK,
)
def delete_exam_reviewer(
    reviewer_id: uuid.UUID,
    current_user: CurrentUser,
    service: ExamServiceDep,
) -> ApiResponse[ExamDeleteResponse]:
    result = service.delete_reviewer(current_user.id, reviewer_id)
    return ApiResponse(data=result, message="Exam reviewer deleted")


@exams_router.post(
    "/{reviewer_id}/attempts",
    response_model=ApiResponse[ExamAttemptStartResponse],
    status_code=status.HTTP_201_CREATED,
)
def start_exam_attempt(
    reviewer_id: uuid.UUID,
    current_user: CurrentUser,
    service: ExamServiceDep,
) -> ApiResponse[ExamAttemptStartResponse]:
    """
    Starts an exam attempt session. Returns student-safe questions (no answer keys or explanations).
    """
    attempt_res = service.start_attempt(current_user.id, reviewer_id)
    return ApiResponse(data=attempt_res, message="Exam attempt started")


@exams_router.get(
    "/attempts/{attempt_id}",
    response_model=ApiResponse[ExamAttemptStartResponse],
    status_code=status.HTTP_200_OK,
)
def get_exam_attempt(
    attempt_id: uuid.UUID,
    current_user: CurrentUser,
    service: ExamServiceDep,
) -> ApiResponse[ExamAttemptStartResponse]:
    """
    Retrieves in-progress attempt masked questions for resuming an exam attempt.
    """
    attempt_res = service.get_attempt(current_user.id, attempt_id)
    return ApiResponse(data=attempt_res)


@exams_router.post(
    "/attempts/{attempt_id}/submit",
    response_model=ApiResponse[ExamResultRead],
    status_code=status.HTTP_200_OK,
)
@exams_router.post(
    "/{reviewer_id}/attempts/{attempt_id}/submit",
    response_model=ApiResponse[ExamResultRead],
    status_code=status.HTTP_200_OK,
)
def submit_exam_attempt(
    attempt_id: uuid.UUID,
    payload: ExamAttemptSubmitRequest,
    current_user: CurrentUser,
    service: ExamServiceDep,
    reviewer_id: uuid.UUID | None = None,
) -> ApiResponse[ExamResultRead]:
    """
    Submits student answers for deterministic and AI-assisted grading.
    Supports both top-level and nested attempt routes.
    """
    result = service.submit_attempt(current_user.id, attempt_id, payload)
    return ApiResponse(data=result, message="Exam submitted and graded")


@exams_router.get(
    "/attempts/{attempt_id}/results",
    response_model=ApiResponse[ExamResultRead],
    status_code=status.HTTP_200_OK,
)
@exams_router.get(
    "/{reviewer_id}/attempts/{attempt_id}/results",
    response_model=ApiResponse[ExamResultRead],
    status_code=status.HTTP_200_OK,
)
def get_attempt_results(
    attempt_id: uuid.UUID,
    current_user: CurrentUser,
    service: ExamServiceDep,
    reviewer_id: uuid.UUID | None = None,
) -> ApiResponse[ExamResultRead]:
    """
    Retrieves full graded results with explanations, feedback, and source references.
    Supports both top-level and nested attempt routes.
    """
    result = service.get_attempt_results(current_user.id, attempt_id)
    return ApiResponse(data=result)
