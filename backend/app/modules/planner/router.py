import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.planner.models import PlannerTask
from app.modules.planner.schemas import (
    PlannerTaskCreate,
    PlannerTaskDeleteResponse,
    PlannerTaskListResponse,
    PlannerTaskRead,
    PlannerTaskUpdate,
)
from app.shared.responses import ApiResponse

planner_router = APIRouter(prefix="/planner", tags=["Planner"])


@planner_router.get(
    "/tasks",
    response_model=ApiResponse[PlannerTaskListResponse],
)
def list_tasks(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> ApiResponse[PlannerTaskListResponse]:
    stmt = (
        select(PlannerTask)
        .where(PlannerTask.user_id == current_user.id)
        .order_by(PlannerTask.created_at.desc())
    )
    if date is not None:
        stmt = stmt.where(PlannerTask.task_date == date)

    tasks = list(db.scalars(stmt).all())
    return ApiResponse(
        data=PlannerTaskListResponse(
            items=[PlannerTaskRead.model_validate(t) for t in tasks],
            total=len(tasks),
        )
    )


@planner_router.post(
    "/tasks",
    response_model=ApiResponse[PlannerTaskRead],
    status_code=status.HTTP_201_CREATED,
)
def create_task(
    payload: PlannerTaskCreate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[PlannerTaskRead]:
    task = PlannerTask(
        user_id=current_user.id,
        title=payload.title.strip(),
        category=payload.category,
        task_date=payload.date,
        time_label=payload.time_label.strip() or "10:00 AM",
        duration_minutes=payload.duration_minutes,
        completed=False,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return ApiResponse(data=PlannerTaskRead.model_validate(task))


@planner_router.patch(
    "/tasks/{task_id}",
    response_model=ApiResponse[PlannerTaskRead],
)
def update_task(
    task_id: uuid.UUID,
    payload: PlannerTaskUpdate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[PlannerTaskRead]:
    task = db.scalar(
        select(PlannerTask).where(
            PlannerTask.id == task_id,
            PlannerTask.user_id == current_user.id,
        )
    )
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    if payload.title is not None:
        task.title = payload.title.strip()
    if payload.category is not None:
        task.category = payload.category
    if payload.date is not None:
        task.task_date = payload.date
    if payload.time_label is not None:
        task.time_label = payload.time_label.strip() or "10:00 AM"
    if payload.duration_minutes is not None:
        task.duration_minutes = payload.duration_minutes
    if payload.completed is not None:
        task.completed = payload.completed

    db.commit()
    db.refresh(task)
    return ApiResponse(data=PlannerTaskRead.model_validate(task))


@planner_router.delete(
    "/tasks/{task_id}",
    response_model=ApiResponse[PlannerTaskDeleteResponse],
)
def delete_task(
    task_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[PlannerTaskDeleteResponse]:
    task = db.scalar(
        select(PlannerTask).where(
            PlannerTask.id == task_id,
            PlannerTask.user_id == current_user.id,
        )
    )
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    db.delete(task)
    db.commit()
    return ApiResponse(data=PlannerTaskDeleteResponse(deleted=True, id=task_id))
