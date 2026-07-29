import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database.session import get_db_session
from app.health.schemas import HealthResponse, PublicHealthResponse

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)

SettingsDependency = Annotated[Settings, Depends(get_settings)]
SessionDependency = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("", response_model=PublicHealthResponse)
async def health() -> PublicHealthResponse:
    """Report API process health without requiring database connectivity."""

    return PublicHealthResponse(status="ok")


@router.get("/database", response_model=HealthResponse)
async def database_health(
    settings: SettingsDependency,
    session: SessionDependency,
) -> HealthResponse | JSONResponse:
    """Report PostgreSQL connectivity independently from process health."""

    try:
        await session.execute(text("SELECT 1"))
    except (OSError, SQLAlchemyError):
        logger.warning("Database health check failed", exc_info=True)
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="unavailable",
                service=settings.service_name,
                version=settings.version,
            ).model_dump(),
        )

    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.version,
    )
