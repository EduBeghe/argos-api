from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.dashboard.schemas import DashboardOverviewResponse
from app.dashboard.service import build_dashboard_overview
from app.paper.portfolio import PaperPortfolioValuationError

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
Session = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("/overview", response_model=DashboardOverviewResponse)
async def dashboard_overview(session: Session) -> DashboardOverviewResponse:
    try:
        return await build_dashboard_overview(
            session=session, generated_at=datetime.now(UTC)
        )
    except PaperPortfolioValuationError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
