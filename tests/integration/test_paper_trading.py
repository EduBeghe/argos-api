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
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from app.paper.calculation import PaperExecutionError
from app.paper.cycle_orchestration import run_instrument_cycle
from app.paper.decision_cycles import (
    DecisionCycleStartDelayError,
    DecisionEvidenceUnavailableError,
    Instrument,
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
    PersistedPaperPortfolioStateSource,
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
    await session.refresh(await session.get(PaperTradingRunRecord, run.id))
    persisted = await session.get(PaperTradingRunRecord, run.id)
    assert persisted is not None and persisted.current_cash == Decimal(
        "10000.000000000000000000"
    )
    assert await session.scalar(select(PaperPositionRecord.id)) is None
    assert await session.scalar(select(PaperTradeRecord.id)) is None


@pytest.mark.parametrize(
    ("run_start", "decision_offset", "executed_at", "message"),
    [
        (NOW, -10, NOW + timedelta(minutes=1), "predates the active"),
        (NOW, 0, NOW + timedelta(minutes=1), "predates the Decision"),
        (
            NOW + timedelta(minutes=10),
            10,
            NOW + timedelta(minutes=9),
            "predates the active run",
        ),
    ],
)
async def test_experiment_boundary_rejection_leaves_no_mutation(
    session: AsyncSession,
    run_start: datetime,
    decision_offset: int,
    executed_at: datetime,
    message: str,
) -> None:
    run = await reset_paper_account(session=session, started_at=run_start)
    await session.commit()
    decision = await _decision(
        session,
        score_value=Decimal("0.60"),
        position_is_open=False,
        offset=decision_offset,
    )
    with pytest.raises(PaperExecutionError, match=message):
        await execute_decision(
            session=session, decision=decision, executed_at=executed_at
        )
    await session.rollback()
    persisted = await session.get(PaperTradingRunRecord, run.id)
    assert persisted is not None
    assert persisted.current_cash == Decimal("10000.000000000000000000")
    assert await session.scalar(select(PaperPositionRecord.id)) is None
    assert await session.scalar(select(PaperTradeRecord.id)) is None


async def test_decision_from_stopped_run_cannot_cross_into_successor(
    session: AsyncSession,
) -> None:
    first = await reset_paper_account(session=session, started_at=NOW)
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    await session.commit()
    successor = await reset_paper_account(
        session=session, started_at=NOW + timedelta(hours=1)
    )
    await session.commit()

    with pytest.raises(PaperExecutionError, match="predates the active"):
        await execute_decision(
            session=session,
            decision=decision,
            executed_at=NOW + timedelta(hours=1, minutes=1),
        )
    await session.rollback()
    stopped = await session.get(PaperTradingRunRecord, first.id)
    active = await session.get(PaperTradingRunRecord, successor.id)
    assert stopped is not None and stopped.status == "stopped"
    assert active is not None and active.current_cash == Decimal(
        "10000.000000000000000000"
    )
    assert await session.scalar(select(PaperTradeRecord.id)) is None


async def test_decision_during_active_run_executes_normally(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    trade = await execute_decision(
        session=session,
        decision=decision,
        executed_at=decision.decision_timestamp + timedelta(seconds=1),
    )
    assert isinstance(trade, PaperTradeRecord)


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
        await execute_decision(
            session=session,
            decision=decision,
            executed_at=decision.decision_timestamp + timedelta(seconds=1),
        )
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
    ).model_copy(update={"close_time": NOW, "ingested_at": PAPER_CANDLE_MARKER})
    session.add(to_record(candle))
    decision = await _decision(
        session, score_value=Decimal("0.60"), position_is_open=False
    )
    trade = await execute_decision(
        session=session,
        decision=decision,
        executed_at=decision.decision_timestamp + timedelta(seconds=1),
    )
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


async def test_historical_queries_return_stopped_run_positions_trades_and_pages(
    session: AsyncSession,
) -> None:
    first = await reset_paper_account(session=session, started_at=NOW)
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
    second = await reset_paper_account(
        session=session, started_at=NOW + timedelta(hours=1)
    )
    await session.commit()

    runs = await list_runs(session, limit=10, offset=0)
    page = await list_runs(session, limit=1, offset=1)
    historical = await get_run(session, run_id=first.id)
    positions = await list_positions(session, run_id=first.id, limit=10, offset=0)
    trades = await list_trades(session, run_id=first.id, limit=10, offset=0)
    assert [item.id for item in runs] == [second.id, first.id]
    assert [item.id for item in page] == [first.id]
    assert historical is not None and historical.status == "stopped"
    assert len(positions) == 1 and positions[0].status == PaperPositionStatus.OPEN
    assert len(trades) == 1 and trades[0].id == trade.id


async def test_boundary_evidence_rejects_future_missing_and_stale_scores(
    session: AsyncSession,
) -> None:
    instrument = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    boundary = NOW + timedelta(minutes=30)
    with pytest.raises(DecisionEvidenceUnavailableError, match="not found"):
        await get_boundary_score_evidence(
            session=session,
            instrument=instrument,
            boundary=boundary,
            run_started_at=NOW,
        )
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=boundary + timedelta(minutes=30),
    )
    with pytest.raises(DecisionEvidenceUnavailableError, match="not found"):
        await get_boundary_score_evidence(
            session=session,
            instrument=instrument,
            boundary=boundary,
            run_started_at=NOW,
        )
    await session.rollback()
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=boundary - timedelta(minutes=15),
    )
    with pytest.raises(DecisionEvidenceUnavailableError, match="stale"):
        await get_boundary_score_evidence(
            session=session,
            instrument=instrument,
            boundary=boundary,
            run_started_at=NOW,
        )


async def test_decision_cycle_open_retry_and_no_action_are_auditable(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    boundary = NOW + timedelta(minutes=30)
    btc = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    eth = Instrument(
        Asset.ETH_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=boundary,
    )
    await _cycle_evidence(
        session,
        asset=Asset.ETH_USDT,
        score_value=Decimal("0.30"),
        boundary=boundary,
    )
    opened = await run_instrument_cycle(
        session=session,
        instrument=btc,
        boundary=boundary,
        execution_time=boundary + timedelta(minutes=2),
    )
    no_action = await run_instrument_cycle(
        session=session,
        instrument=eth,
        boundary=boundary,
        execution_time=boundary + timedelta(minutes=2),
    )
    await session.commit()
    retry = await run_instrument_cycle(
        session=session,
        instrument=btc,
        boundary=boundary,
        execution_time=boundary + timedelta(minutes=3),
    )
    assert opened.trade_id is not None and opened.action == "open_position"
    assert no_action.trade_id is None and no_action.action == "no_action"
    assert retry.skipped is True and retry.cycle_id == opened.cycle_id
    assert len((await session.scalars(select(PaperDecisionCycleRecord.id))).all()) == 2


async def test_concurrent_cycle_claim_executes_once(session: AsyncSession) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    boundary = NOW + timedelta(minutes=30)
    instrument = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=boundary,
    )
    await session.commit()
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def attempt() -> object:
        async with factory() as concurrent_session:
            result = await run_instrument_cycle(
                session=concurrent_session,
                instrument=instrument,
                boundary=boundary,
                execution_time=boundary + timedelta(minutes=2),
            )
            await concurrent_session.commit()
            return result

    outcomes = await asyncio.gather(attempt(), attempt(), return_exceptions=True)
    assert all(not isinstance(outcome, BaseException) for outcome in outcomes)
    async with factory() as verification:
        assert (
            len((await verification.scalars(select(PaperDecisionCycleRecord.id))).all())
            == 1
        )
        assert len((await verification.scalars(select(PaperTradeRecord.id))).all()) == 1
        assert (
            len((await verification.scalars(select(PaperPositionRecord.id))).all()) == 1
        )
        persisted_run = await verification.get(PaperTradingRunRecord, run.id)
        trade = await verification.scalar(select(PaperTradeRecord))
        assert persisted_run is not None and trade is not None
        assert persisted_run.current_cash == Decimal("10000") + trade.cash_impact
    await engine.dispose()


async def test_failed_cycle_claim_rolls_back_and_retry_recovers(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    await session.commit()
    boundary = NOW + timedelta(minutes=30)
    instrument = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    with pytest.raises(DecisionEvidenceUnavailableError):
        await run_instrument_cycle(
            session=session,
            instrument=instrument,
            boundary=boundary,
            execution_time=boundary + timedelta(minutes=2),
        )
    await session.rollback()
    assert await session.scalar(select(PaperDecisionCycleRecord.id)) is None
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=boundary,
    )
    recovered = await run_instrument_cycle(
        session=session,
        instrument=instrument,
        boundary=boundary,
        execution_time=boundary + timedelta(minutes=3),
    )
    assert recovered.action == "open_position"


@pytest.mark.parametrize(
    ("delay_minutes", "message"),
    [(-1, "must not precede"), (11, "maximum delay"), (29, "maximum delay")],
)
async def test_delayed_cycle_rejection_leaves_no_persisted_mutation(
    session: AsyncSession, delay_minutes: int, message: str
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    await session.commit()
    boundary = NOW + timedelta(minutes=30)
    instrument = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=boundary,
    )
    await session.commit()
    with pytest.raises(DecisionCycleStartDelayError, match=message):
        await run_instrument_cycle(
            session=session,
            instrument=instrument,
            boundary=boundary,
            execution_time=boundary + timedelta(minutes=delay_minutes),
        )
    await session.rollback()
    persisted_run = await session.get(PaperTradingRunRecord, run.id)
    assert persisted_run is not None and persisted_run.current_cash == Decimal("10000")
    assert await session.scalar(select(PaperDecisionCycleRecord.id)) is None
    assert await session.scalar(select(DecisionSnapshotRecord.id)) is None
    assert await session.scalar(select(PaperTradeRecord.id)) is None
    assert await session.scalar(select(PaperPositionRecord.id)) is None


async def test_cycle_start_delay_accepts_inclusive_ten_minutes(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    boundary = NOW + timedelta(minutes=30)
    instrument = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=boundary,
    )
    result = await run_instrument_cycle(
        session=session,
        instrument=instrument,
        boundary=boundary,
        execution_time=boundary + timedelta(minutes=10),
    )
    assert result.action == "open_position"


async def test_cycle_after_delayed_rejection_uses_next_boundary(
    session: AsyncSession,
) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    instrument = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    delayed_boundary = NOW + timedelta(minutes=30)
    with pytest.raises(DecisionCycleStartDelayError):
        await run_instrument_cycle(
            session=session,
            instrument=instrument,
            boundary=delayed_boundary,
            execution_time=delayed_boundary + timedelta(minutes=11),
        )
    next_boundary = delayed_boundary + timedelta(minutes=30)
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=next_boundary,
    )
    result = await run_instrument_cycle(
        session=session,
        instrument=instrument,
        boundary=next_boundary,
        execution_time=next_boundary + timedelta(minutes=2),
    )
    assert result.boundary == next_boundary and result.action == "open_position"


async def test_following_cycle_executes_sell_all(session: AsyncSession) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    instrument = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    open_boundary = NOW + timedelta(minutes=30)
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=open_boundary,
    )
    opened = await run_instrument_cycle(
        session=session,
        instrument=instrument,
        boundary=open_boundary,
        execution_time=open_boundary + timedelta(minutes=2),
    )
    await session.commit()
    sell_boundary = open_boundary + timedelta(minutes=30)
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.19"),
        boundary=sell_boundary,
    )
    sold = await run_instrument_cycle(
        session=session,
        instrument=instrument,
        boundary=sell_boundary,
        execution_time=sell_boundary + timedelta(minutes=2),
    )
    await session.commit()
    retry = await run_instrument_cycle(
        session=session,
        instrument=instrument,
        boundary=sell_boundary,
        execution_time=sell_boundary + timedelta(minutes=3),
    )
    position = await session.scalar(select(PaperPositionRecord))
    trades = list(
        (
            await session.scalars(
                select(PaperTradeRecord).order_by(PaperTradeRecord.executed_at)
            )
        ).all()
    )
    run = await get_active_run(session, currency="USDT")
    sell_cycle = await session.get(PaperDecisionCycleRecord, sold.cycle_id)
    assert opened.action == "open_position" and opened.trade_id is not None
    assert sold.action == "sell_all" and sold.trade_id is not None
    assert position is not None and position.status == PaperPositionStatus.CLOSED.value
    assert position.quantity == 0
    assert len(trades) == 2 and trades[1].side == "sell"
    assert trades[1].quantity == trades[0].quantity
    assert run is not None
    assert (
        run.current_cash
        == Decimal("10000") + trades[0].cash_impact + trades[1].cash_impact
    )
    assert sell_cycle is not None
    assert sell_cycle.result == "sell_all"
    assert sell_cycle.decision_snapshot_id == sold.decision_snapshot_id
    assert sell_cycle.paper_trade_id == sold.trade_id
    assert retry.skipped is True and retry.cycle_id == sold.cycle_id
    assert len((await session.scalars(select(PaperTradeRecord.id))).all()) == 2


async def test_paper_decision_cycle_database_constraints(session: AsyncSession) -> None:
    await reset_paper_account(session=session, started_at=NOW)
    btc = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    eth = Instrument(
        Asset.ETH_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    open_boundary = NOW + timedelta(minutes=30)
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.60"),
        boundary=open_boundary,
    )
    await _cycle_evidence(
        session,
        asset=Asset.ETH_USDT,
        score_value=Decimal("0.30"),
        boundary=open_boundary,
    )
    opened = await run_instrument_cycle(
        session=session,
        instrument=btc,
        boundary=open_boundary,
        execution_time=open_boundary + timedelta(minutes=2),
    )
    no_action = await run_instrument_cycle(
        session=session,
        instrument=eth,
        boundary=open_boundary,
        execution_time=open_boundary + timedelta(minutes=2),
    )
    sell_boundary = open_boundary + timedelta(minutes=30)
    await _cycle_evidence(
        session,
        asset=Asset.BTC_USDT,
        score_value=Decimal("0.19"),
        boundary=sell_boundary,
    )
    sold = await run_instrument_cycle(
        session=session,
        instrument=btc,
        boundary=sell_boundary,
        execution_time=sell_boundary + timedelta(minutes=2),
    )
    await session.commit()

    async def rejects(cycle_id: object, **values: object) -> None:
        with pytest.raises(IntegrityError):
            await session.execute(
                update(PaperDecisionCycleRecord)
                .where(PaperDecisionCycleRecord.id == cycle_id)
                .values(**values)
            )
            await session.flush()
        await session.rollback()

    await rejects(opened.cycle_id, result="arbitrary")
    await rejects(opened.cycle_id, paper_trade_id=None)
    await rejects(sold.cycle_id, paper_trade_id=None)
    await rejects(
        opened.cycle_id,
        status="processing",
        completed_at=None,
        result=None,
    )
    await rejects(
        sold.cycle_id,
        decision_snapshot_id=opened.decision_snapshot_id,
    )
    await rejects(sold.cycle_id, paper_trade_id=opened.trade_id)
    await rejects(
        opened.cycle_id,
        started_at=open_boundary - timedelta(microseconds=1),
    )
    await rejects(
        opened.cycle_id,
        completed_at=open_boundary + timedelta(minutes=1),
    )

    # Release a Trade from its original cycle without violating lifecycle, then prove
    # that a completed NO_ACTION cycle still cannot reference that Trade.
    await session.execute(
        update(PaperDecisionCycleRecord)
        .where(PaperDecisionCycleRecord.id == opened.cycle_id)
        .values(result="no_action", paper_trade_id=None)
    )
    with pytest.raises(IntegrityError):
        await session.execute(
            update(PaperDecisionCycleRecord)
            .where(PaperDecisionCycleRecord.id == no_action.cycle_id)
            .values(paper_trade_id=opened.trade_id)
        )
        await session.flush()
    await session.rollback()

