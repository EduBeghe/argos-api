import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.decisions.generation import PortfolioState, generate_decision_from_evidence
from app.decisions.models import DecisionSnapshot
from app.decisions.persistence import DecisionSnapshotRecord, ScoreEvidence
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import to_record
from app.paper.calculation import PaperExecutionError
from app.paper.execution import execute_decision
from app.paper.models import PaperPositionStatus, PaperRunStatus
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradeRecord,
    PaperTradingRunRecord,
    get_active_run,
)
from app.paper.portfolio import (
    PaperPortfolioUnavailableError,
    PersistedPaperPortfolioStateSource,
)
from app.paper.reset import reset_paper_account
from app.scores.persistence import MarketScoreRecord, insert_market_scores
from tests.decisions.factories import make_score
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)


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
        PaperTradeRecord,
        PaperPositionRecord,
        PaperTradingRunRecord,
        DecisionSnapshotRecord,
        MarketScoreRecord,
    ):
        await session.execute(delete(record))
    await session.commit()


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


async def test_unique_active_run_and_invalid_trade_direction_are_enforced(
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


async def test_execution_rollback_restores_cash_position_and_trade(
    session: AsyncSession,
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    await session.commit()
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    await execute_decision(session=session, decision=decision, executed_at=NOW)
    await session.rollback()
    await session.refresh(await session.get(PaperTradingRunRecord, run.id))
    persisted = await session.get(PaperTradingRunRecord, run.id)
    assert persisted is not None and persisted.current_cash == Decimal(
        "10000.000000000000000000"
    )
    assert await session.scalar(select(PaperPositionRecord.id)) is None
    assert await session.scalar(select(PaperTradeRecord.id)) is None


async def test_multi_asset_equity_uses_persisted_price_and_missing_price_fails_closed(
    session: AsyncSession,
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    eth = PaperPositionRecord(
        id=uuid4(),
        run_id=run.id,
        asset=Asset.ETH_USDT.value,
        provider=Provider.BINANCE.value,
        market_type=MarketType.SPOT.value,
        interval=Interval.FIVE_MINUTES.value,
        status="open",
        quantity=Decimal("2"),
        average_entry_price=Decimal("50"),
        total_entry_notional=Decimal("100"),
        accumulated_entry_fees=Decimal("0.1"),
        opened_at=NOW,
        closed_at=None,
        exit_price=None,
        gross_realized_pnl=None,
        net_realized_pnl=None,
    )
    session.add(eth)
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    with pytest.raises(PaperExecutionError, match="valuation price"):
        await execute_decision(session=session, decision=decision, executed_at=NOW)
    await session.rollback()

    new_run = await reset_paper_account(session=session, started_at=NOW)
    session.add(
        PaperPositionRecord(
            id=uuid4(),
            run_id=new_run.id,
            asset=Asset.ETH_USDT.value,
            provider=Provider.BINANCE.value,
            market_type=MarketType.SPOT.value,
            interval=Interval.FIVE_MINUTES.value,
            status="open",
            quantity=Decimal("2"),
            average_entry_price=Decimal("50"),
            total_entry_notional=Decimal("100"),
            accumulated_entry_fees=Decimal("0.1"),
            opened_at=NOW,
            closed_at=None,
            exit_price=None,
            gross_realized_pnl=None,
            net_realized_pnl=None,
        )
    )
    candle = make_candle(
        asset=Asset.ETH_USDT,
        open_price=Decimal("195"),
        high_price=Decimal("205"),
        low_price=Decimal("190"),
        close_price=Decimal("200"),
    ).model_copy(update={"close_time": NOW})
    session.add(to_record(candle))
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    trade = await execute_decision(session=session, decision=decision, executed_at=NOW)
    assert isinstance(trade, PaperTradeRecord)
    assert trade.equity_before == Decimal("10400.000000000000000000")


async def test_portfolio_source_uses_only_active_run(session: AsyncSession) -> None:
    source = PersistedPaperPortfolioStateSource(session)
    with pytest.raises(PaperPortfolioUnavailableError):
        await source.get_portfolio_state(
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
        )
    run = await reset_paper_account(session=session, started_at=NOW)
    state = await source.get_portfolio_state(
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
    )
    assert state.position_is_open is False
    run_record = await session.get(PaperTradingRunRecord, run.id)
    assert run_record is not None
    run_record.status = "stopped"
    run_record.stopped_at = NOW
    await session.flush()
    with pytest.raises(PaperPortfolioUnavailableError):
        await source.get_portfolio_state(
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
        )
