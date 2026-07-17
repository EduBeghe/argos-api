from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.models import PaperPositionStatus
from app.paper.persistence import (
    PaperTradingRunRecord,
    get_active_run,
    list_positions,
    list_trades,
    run_domain,
)
from app.paper.schemas import (
    PaperAccountResponse,
    PaperPositionResponse,
    PaperTradeResponse,
)

router = APIRouter(prefix="/paper", tags=["paper"])
Session = Annotated[AsyncSession, Depends(get_db_session)]


async def _active(session: AsyncSession) -> PaperTradingRunRecord:
    record = await get_active_run(session, currency=DEFAULT_ACCOUNT_CURRENCY)
    if record is None:
        raise HTTPException(
            status_code=404, detail="Active Paper Trading run not found"
        )
    return record


@router.get("/account", response_model=PaperAccountResponse)
async def paper_account(session: Session) -> PaperAccountResponse:
    return PaperAccountResponse.from_domain(run_domain(await _active(session)))


@router.get("/positions", response_model=list[PaperPositionResponse])
async def paper_positions(
    session: Session, status: Annotated[PaperPositionStatus | None, Query()] = None
) -> list[PaperPositionResponse]:
    run = await _active(session)
    return [
        PaperPositionResponse.model_validate(item)
        for item in await list_positions(session, run_id=run.id, status=status)
    ]


@router.get("/trades", response_model=list[PaperTradeResponse])
async def paper_trades(
    session: Session,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PaperTradeResponse]:
    run = await _active(session)
    return [
        PaperTradeResponse.from_record(item)
        for item in await list_trades(
            session, run_id=run.id, limit=limit, offset=offset
        )
    ]
