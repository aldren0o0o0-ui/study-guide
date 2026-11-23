import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskCategory = Literal["Study", "Work", "Personal", "Project", "General"]


class PlannerTaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    category: TaskCategory = "Study"
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    time_label: str = Field(default="10:00 AM", min_length=1, max_length=32)
    duration_minutes: int = Field(default=45, ge=5, le=720)


class PlannerTaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    category: TaskCategory | None = None
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    time_label: str | None = Field(default=None, min_length=1, max_length=32)
    duration_minutes: int | None = Field(default=None, ge=5, le=720)
    completed: bool | None = None


class PlannerTaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    category: str
    task_date: str
    time_label: str
    duration_minutes: int
    completed: bool
    created_at: datetime
    updated_at: datetime


class PlannerTaskListResponse(BaseModel):
    items: list[PlannerTaskRead]
    total: int


class PlannerTaskDeleteResponse(BaseModel):
    deleted: bool
    id: uuid.UUID

