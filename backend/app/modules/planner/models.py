import uuid

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class PlannerTask(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "planner_tasks"
    __table_args__ = (
        Index("ix_planner_tasks_user_id_date", "user_id", "task_date"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False, default="Study")
    task_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    time_label: Mapped[str] = mapped_column(String(32), nullable=False, default="10:00 AM")
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=45)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

