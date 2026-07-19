import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.decisions.generation import PortfolioState, generate_decision_from_evidence
from app.decisions.models import DecisionSnapshot
from app.decisions.persistence import DecisionSnapshotRecord, ScoreEvidence
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from app.paper.calculation import PaperExecutionError
from app.paper.cycle_orchestration import run_instrument_cycle
from app.paper.decision_cycles import (
    DecisionCycleStartDelayError,
    DecisionEvidenceUnavailableError,
    PaperDecisionCycleRecord,
    get_boundary_score_evidence,
)
from app.paper.execution import execute_decision
from app.paper.models import PaperPositionStatus, PaperRunStatus
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradeRecord,
    PaperTradingRunRecord,
    get_active_run,
    get_run,
    list_positions,
    list_runs,
    list_trades,
)
from app.paper.portfolio import (
    PaperPortfolioUnavailableError,
    PaperPortfolioValuationError,
    PersistedPaperPortfolioStateSource,
    value_paper_run,
)
from app.paper.reset import reset_paper_account
from app.scores.persistence import MarketScoreRecord, insert_market_scores
from tests.decisions.factories import make_score
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)
PAPER_CANDLE_MARKER = datetime(2099, 1, 1, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        await _clear(database_session)
        yield database_session
        await database_session.rollback()
        await _clear(database_session)
    await engine.dispose()


async def _clear(session: AsyncSession) -> None:
    for record in (
        PaperDecisionCycleRecord,
        PaperTradeRecord,
        PaperPositionRecord,
        PaperTradingRunRecord,
        DecisionSnapshotRecord,
        MarketScoreRecord,
    ):
        await session.execute(delete(record))
    await session.execute(
        delete(MarketCandleRecord).where(
            MarketCandleRecord.ingested_at == PAPER_CANDLE_MARKER
        )
    )
    await session.commit()


async def _cycle_evidence(
    session: AsyncSession,
    *,
    asset: Asset,
    score_value: Decimal,
    boundary: datetime,
    close_offset: timedelta = timedelta(microseconds=-1000),
) -> None:
    open_time = boundary - timedelta(minutes=5)
    candle = make_candle(
        asset=asset,
        open_time=open_time,
        close_time=boundary + close_offset,
        ingested_at=PAPER_CANDLE_MARKER,
    )
    session.add(to_record(candle))
    await session.flush()
    score = make_score(overall_score=score_value).model_copy(
        update={"id": uuid4(), "asset": asset, "candle_open_time": open_time}
    )
    assert await insert_market_scores(session, [score]) == 1


async def _decision(
    session: AsyncSession,
    *,
    score_value: Decimal,
    position_is_open: bool,
    asset: Asset = Asset.BTC_USDT,
    offset: int = 0,
) -> DecisionSnapshot:
    score = make_score(overall_score=score_value).model_copy(
        update={
            "id": uuid4(),
            "asset": asset,
            "candle_open_time": NOW + timedelta(minutes=offset),
        }
    )
    assert await insert_market_scores(session, [score]) == 1
    result = await generate_decision_from_evidence(
        session=session,
        evidence=ScoreEvidence(score=score, source_market_price=Decimal("100")),
        portfolio_state=PortfolioState(position_is_open=position_is_open),
        decision_timestamp=NOW + timedelta(minutes=offset + 5),
    )
    return result.snapshot


async def test_reset_stops_run_preserves_open_position_and_starts_empty_successor(
    session: AsyncSession,
) -> None:
    first = await reset_paper_account(session=session, started_at=NOW)
    position = PaperPositionRecord(
        id=uuid4(),
        run_id=first.id,
        asset=Asset.ETH_USDT.value,
        provider=Provider.BINANCE.value,
        market_type=MarketType.SPOT.value,
        interval=Interval.FIVE_MINUTES.value,
        status=PaperPositionStatus.OPEN.value,
        quantity=Decimal("2"),
        average_entry_price=Decimal("100"),
        total_entry_notional=Decimal("200"),
        accumulated_entry_fees=Decimal("0.2"),
        opened_at=NOW,
        closed_at=None,
        exit_price=None,
        gross_realized_pnl=None,
        net_realized_pnl=None,
    )
    session.add(position)
    await session.commit()

    successor = await reset_paper_account(
        session=session,
        started_at=NOW + timedelta(hours=1),
        initial_cash=Decimal("12000"),
        reason="new baseline",
    )
    await session.commit()

    stopped = await session.get(PaperTradingRunRecord, first.id)
    preserved = await session.get(PaperPositionRecord, position.id)
    assert stopped is not None and stopped.status == PaperRunStatus.STOPPED.value
    assert stopped.current_cash == Decimal("10000.000000000000000000")
    assert stopped.stop_reason == "new baseline"
    assert preserved is not None and preserved.status == PaperPositionStatus.OPEN.value
    assert preserved.quantity == Decimal("2.000000000000000000")
    assert successor.predecessor_run_id == first.id
    assert successor.current_cash == Decimal("12000.000000000000000000")
    assert await session.scalar(select(PaperTradeRecord.id)) is None
    assert (
        await session.scalar(
            select(PaperPositionRecord.id).where(
                PaperPositionRecord.run_id == successor.id
            )
        )
        is None
    )


async def test_reset_rollback_restores_previous_active_run(
    session: AsyncSession,
) -> None:
    first = await reset_paper_account(session=session, started_at=NOW)
    await session.commit()
    await reset_paper_account(session=session, started_at=NOW + timedelta(hours=1))
    await session.rollback()
    active = await get_active_run(session, currency="USDT")
    assert active is not None and active.id == first.id


async def test_concurrent_resets_leave_exactly_one_active_run(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    await session.commit()
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def reset_at(moment: datetime) -> None:
        async with factory() as concurrent_session:
            await reset_paper_account(session=concurrent_session, started_at=moment)
            await concurrent_session.commit()

    outcomes = await asyncio.gather(
        reset_at(NOW + timedelta(hours=1)),
        reset_at(NOW + timedelta(hours=2)),
        return_exceptions=True,
    )
    assert sum(isinstance(outcome, IntegrityError) for outcome in outcomes) <= 1
    assert sum(outcome is None for outcome in outcomes) >= 1
    async with factory() as verification:
        active_ids = (
            await verification.scalars(
                select(PaperTradingRunRecord.id).where(
                    PaperTradingRunRecord.status == PaperRunStatus.ACTIVE.value
                )
            )
        ).all()
        assert len(active_ids) == 1
    await engine.dispose()


async def test_unique_active_run_is_enforced(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    await session.commit()
    duplicate = PaperTradingRunRecord(
        id=uuid4(),
        status="active",
        account_currency="USDT",
        initial_cash=Decimal("1"),
        current_cash=Decimal("1"),
        configuration_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=NOW,
        stopped_at=None,
        stop_reason=None,
        predecessor_run_id=None,
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_database_rejects_invalid_run_position_and_trade_rows(
    session: AsyncSession,
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    trade = await execute_decision(
        session=session,
        decision=decision,
        executed_at=decision.decision_timestamp + timedelta(seconds=1),
    )
    assert isinstance(trade, PaperTradeRecord)
    await session.commit()
    trade_id = trade.id
    position_id = trade.position_id
    run_id = run.id

    async def rejects(record: object, attribute: str, value: object) -> None:
        setattr(record, attribute, value)
        with pytest.raises(IntegrityError):
            await session.flush()
        await session.rollback()

    persisted_trade = await session.get(PaperTradeRecord, trade_id)
    assert persisted_trade is not None
    await rejects(persisted_trade, "cash_impact", Decimal("1"))
    persisted_trade = await session.get(PaperTradeRecord, trade_id)
    assert persisted_trade is not None
    persisted_trade.side = "sell"
    persisted_trade.cash_impact = Decimal("-1")
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()
    persisted_trade = await session.get(PaperTradeRecord, trade_id)
    assert persisted_trade is not None
    await rejects(persisted_trade, "asset", "")
    persisted_trade = await session.get(PaperTradeRecord, trade_id)
    assert persisted_trade is not None
    await rejects(persisted_trade, "configuration_version", 0)

    persisted_run = await session.get(PaperTradingRunRecord, run_id)
    assert persisted_run is not None
    await rejects(persisted_run, "account_currency", "")
    persisted_run = await session.get(PaperTradingRunRecord, run_id)
    assert persisted_run is not None
    await rejects(persisted_run, "configuration_version", 0)
    persisted_run = await session.get(PaperTradingRunRecord, run_id)
    assert persisted_run is not None
    await rejects(persisted_run, "status", "stopped")
    persisted_run = await session.get(PaperTradingRunRecord, run_id)
    assert persisted_run is not None
    await rejects(persisted_run, "stopped_at", NOW + timedelta(hours=1))
    persisted_run = await session.get(PaperTradingRunRecord, run_id)
    assert persisted_run is not None
    await rejects(persisted_run, "predecessor_run_id", persisted_run.id)

    position = await session.get(PaperPositionRecord, position_id)
    assert position is not None
    await rejects(position, "status", "closed")


async def test_open_and_sell_are_atomic_and_duplicate_execution_is_idempotent(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    open_decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    trade = await execute_decision(
        session=session, decision=open_decision, executed_at=NOW + timedelta(minutes=6)
    )
    duplicate = await execute_decision(
        session=session, decision=open_decision, executed_at=NOW + timedelta(minutes=7)
    )
    await session.commit()
    assert isinstance(trade, PaperTradeRecord)
    assert isinstance(duplicate, PaperTradeRecord) and duplicate.id == trade.id
    assert trade.side == "buy" and trade.cash_impact < 0
    position = await session.get(PaperPositionRecord, trade.position_id)
    assert position is not None and position.status == "open"

    sell_decision = await _decision(
        session, score_value=Decimal("0.19"), position_is_open=True, offset=30
    )
    sell = await execute_decision(
        session=session, decision=sell_decision, executed_at=NOW + timedelta(minutes=36)
    )
    await session.commit()
    await session.refresh(position)
    assert isinstance(sell, PaperTradeRecord)
    assert sell.quantity == trade.quantity
    assert sell.side == "sell" and sell.cash_impact > 0
    assert position.status == "closed" and position.quantity == 0
    assert position.gross_realized_pnl is not None
    assert position.net_realized_pnl is not None


async def test_concurrent_execution_mutates_exactly_once(
    session: AsyncSession,
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    await session.commit()
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def attempt() -> object:
        async with factory() as concurrent_session:
            result = await execute_decision(
                session=concurrent_session,
                decision=decision,
                executed_at=decision.decision_timestamp + timedelta(seconds=1),
            )
            await concurrent_session.commit()
            return result

    outcomes = await asyncio.gather(attempt(), attempt(), return_exceptions=True)
    assert all(not isinstance(outcome, BaseException) for outcome in outcomes)
    async with factory() as verification:
        trades = (
            await verification.scalars(
                select(PaperTradeRecord).where(PaperTradeRecord.run_id == run.id)
            )
        ).all()
        positions = (
            await verification.scalars(
                select(PaperPositionRecord).where(PaperPositionRecord.run_id == run.id)
            )
        ).all()
        persisted_run = await verification.get(PaperTradingRunRecord, run.id)
        assert len(trades) == 1
        assert len(positions) == 1
        assert persisted_run is not None
        assert persisted_run.current_cash == Decimal("10000") + trades[0].cash_impact
    await engine.dispose()


async def test_execution_rollback_restores_cash_position_and_trade(
    session: AsyncSession,
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    await session.commit()
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    await execute_decision(
        session=session,
        decision=decision,
        executed_at=decision.decision_timestamp + timedelta(seconds=1),
    )
    await session.rollback()
    await session.refresh(await session.get(PaperTradin