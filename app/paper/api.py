from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database.session import get_db_session
from app.market.catalog import (
    InstrumentCapability,
    MarketCatalogAmbiguityError,
    MarketCatalogError,
    resolve_instrument,
)
from app.market.enums import Asset
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.decision_cycles import get_cycle, list_cycles
from app.paper.equity_snapshots import get_equity_snapshot, list_equity_snapshots
from app.paper.models import PaperPositionStatus
from app.paper.performance import calculate_performance, get_performance_points
from app.paper.persistence import (
    PaperTradingRunRecord,
    get_active_run,
    get_run,
    list_positions,
    list_runs,
    list_trades,
    run_domain,
)
from app.paper.portfolio import PaperPortfolioValuationError, value_paper_run
from app.paper.schemas import (
    PaperAccountResponse,
    PaperDecisionCycleResponse,
    PaperPerformanceResponse,
    PaperPortfolioResponse,
    PaperPortfolioSnapshotResponse,
    PaperPositionResponse,
    PaperTradeQuery,
    PaperTradeResponse,
)

router = APIRouter(prefix="/paper", tags=["paper"])
Session = Annotated[AsyncSession, Depends(get_db_session)]
TradeQuery = Annotated[PaperTradeQuery, Query()]


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


@router.get("/portfolio", response_model=PaperPortfolioResponse)
async def paper_portfolio(
    session: Session,
    run_id: Annotated[UUID | None, Query()] = None,
) -> PaperPortfolioResponse:
    run = await _selected_run(session, run_id)
    try:
        valuation = await value_paper_run(
            session=session, run=run, valued_at=datetime.now(UTC)
        )
    except PaperPortfolioValuationError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return PaperPortfolioResponse.model_validate(valuation)


@router.get("/performance", response_model=PaperPerformanceResponse)
async def paper_performance(
    session: Session,
    run_id: Annotated[UUID | None, Query()] = None,
) -> PaperPerformanceResponse:
    run = await _selected_run(session, run_id)
    performance = calculate_performance(
        run=run_domain(run),
        snapshots=await get_performance_points(session, run_id=run.id),
    )
    return PaperPerformanceResponse.model_validate(
        {
            "run_id": run.id,
            "status": run.status,
            "account_currency": run.account_currency,
            **asdict(performance),
        }
    )


@router.get("/equity-snapshots", response_model=list[PaperPortfolioSnapshotResponse])
async def paper_equity_snapshots(
    session: Session,
    run_id: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
) -> list[PaperPortfolioSnapshotResponse]:
    run = await _selected_run(session, run_id)
    records = await list_equity_snapshots(
        session=session,
        run_id=run.id,
        limit=limit,
        offset=offset,
        order=order,
    )
    return [PaperPortfolioSnapshotResponse.model_validate(record) for record in records]


@router.get(
    "/equity-snapshots/{snapshot_id}", response_model=PaperPortfolioSnapshotResponse
)
async def paper_equity_snapshot(
    snapshot_id: UUID, session: Session
) -> PaperPortfolioSnapshotResponse:
    record = await get_equity_snapshot(session, snapshot_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Paper equity snapshot not found")
    return PaperPortfolioSnapshotResponse.model_validate(record)


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
    params: TradeQuery,
) -> list[PaperTradeResponse]:
    run = await _selected_run(session, params.run_id)
    instrument = None
    if params.asset is not None and params.interval is not None:
        try:
            instrument = resolve_instrument(
                asset=params.asset,
                interval=params.interval,
                capability=InstrumentCapability.DECISION,
                provider=params.provider,
                market_type=params.market_type,
            )
        except MarketCatalogAmbiguityError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except MarketCatalogError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
    return [
        PaperTradeResponse.from_record(item)
        for item in await list_trades(
            session,
            run_id=run.id,
            asset=instrument.asset if instrument else None,
            provider=instrument.provider if instrument else None,
            market_type=instrument.market_type if instrument else None,
            interval=instrument.interval if instrument else None,
            from_time=params.from_time,
            to_time=params.to_time,
            order=params.order,
            limit=params.limit,
            offset=params.offset,
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
