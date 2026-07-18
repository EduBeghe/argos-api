from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.market.enums import Asset
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.decision_cycles import get_cycle, list_cycles
from app.paper.models import PaperPositionStatus
from app.paper.persistence import (
    PaperTradingRunRecord,
    get_active_run,
    get_run,
    list_positions,
    list_runs,
    list_trades,
    run_domain,
)
from app.paper.schemas import (
    PaperAccountResponse,
    PaperDecisionCycleResponse,
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


async def _selected_run(
    session: AsyncSession, run_id: UUID | None
) -> PaperTradingRunRecord:
    if run_id is None:
        return await _active(session)
    record = await get_run(session, run_id=run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Paper Trading run not found")
    return record


@router.get("/account", response_model=PaperAccountResponse)
async def paper_account(session: Session) -> PaperAccountResponse:
    return PaperAccountResponse.from_domain(run_domain(await _active(session)))


@router.get("/runs", response_model=list[PaperAccountResponse])
async def paper_runs(
    session: Session,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PaperAccountResponse]:
    return [
        PaperAccountResponse.from_domain(run)
        for run in await list_runs(session, limit=limit, offset=offset)
    ]


@router.get("/runs/{run_id}", response_model=PaperAccountResponse)
async def paper_run(run_id: UUID, session: Session) -> PaperAccountResponse:
    return PaperAccountResponse.from_domain(
        run_domain(await _selected_run(session, run_id))
    )


@router.get("/positions", response_model=list[PaperPositionResponse])
async def paper_positions(
    session: Session,
    run_id: Annotated[UUID | None, Query()] = None,
    status: Annotated[PaperPositionStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PaperPositionResponse]:
    run = await _selected_run(session, run_id)
    return [
        PaperPositionResponse.model_validate(item)
        for item in await list_positions(
            session,
            run_id=run.id,
            status=status,
            limit=limit,
            offset=offset,
        )
    ]


@router.get("/trades", response_model=list[PaperTradeResponse])
async def paper_trades(
    session: Session,
    run_id: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PaperTradeResponse]:
    run = await _selected_run(session, run_id)
    return [
        PaperTradeResponse.from_record(item)
        for item in await list_trades(
            session, run_id=run.id, limit=limit, offset=offset
        )
    ]


@router.get("/decision-cycles", response_model=list[PaperDecisionCycleResponse])
async def paper_decision_cycles(
    session: Session,
    run_id: Annotated[UUID | None, Query()] = None,
    asset: Annotated[Asset | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PaperDecisionCycleResponse]:
    records = await list_cycles(
        session=session,
        run_id=run_id,
        asset=asset,
        limit=limit,
        offset=offset,
    )
    return [PaperDecisionCycleResponse.model_validate(record) for record in records]


@router.get("/decision-cycles/{cycle_id}", response_model=PaperDecisionCycleResponse)
async def paper_decision_cycle(
    cycle_id: UUID, session: Session
) -> PaperDecisionCycleResponse:
    record = await get_cycle(session, cycle_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Paper Decision Cycle not found")
    return PaperDecisionCycleResponse.model_validate(record)
