from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.config import get_settings
from app.database.session import check_database_status
from app.shared.responses import ApiResponse

router = APIRouter(tags=["Health"])


class HealthStatusData(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    timestamp: datetime
    database: Literal["ok", "unavailable", "not_checked"] = "not_checked"


@router.get("/health", response_model=ApiResponse[HealthStatusData])
def get_health(
    check_db: bool = Query(
        default=False,
        description="When true, checks database connectivity and reports its status separately.",
    ),
) -> ApiResponse[HealthStatusData]:
    """API health check endpoint.

    Always responds independently of database availability. When `check_db=true` is passed,
    database connectivity is probed and reported separately in the `database` field.
    """
    settings = get_settings()
    db_status: Literal["ok", "unavailable", "not_checked"] = (
        check_database_status() if check_db else "not_checked"
    )
    return ApiResponse(
        data=HealthStatusData(
            status="ok",
            service=settings.APP_NAME,
            version=settings.APP_VERSION,
            environment=settings.APP_ENV,
            timestamp=datetime.now(UTC),
            database=db_status,
        )
    )
