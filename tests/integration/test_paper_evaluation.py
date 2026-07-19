from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.decisions.persistence import DecisionSnapshotRecord
from app.market.persistence import MarketCandleRecord
from app.paper.decision_cycles import PaperDecisionCycleRecord
from app.paper.equity_snapshots import (
    PaperPortfolioSnapshotRecord,
    PaperPositionSnapshotRecord,
)
from app.paper.evaluation import MarketIdentity
from app.paper.evaluation_queries import (
    benchmark_candles,
    evaluation_closed_positions,
    evaluation_decisions,
    evaluation_trades,
)
from app.paper.evaluation_service import evaluate_paper_run
from app.paper.performance import get_performance_points
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradeRecord,
    PaperTradingRunRecord,
    get_run,
)
from app.scores.persistence import MarketScoreRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2097, 7, 19, 23, 58, 38, tzinfo=UTC)
BOUNDARY = datetime(2097, 7, 20, 0, 30, tzinfo=UTC)
MARKER = datetime(2097, 7, 20, 1, tzinfo=UTC)
BTC = MarketIdentity("BTC/USDT", "binance", "spot", "5m")
ETH = MarketIdentity("ETH/USDT", "binance", "spot", "5m")


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as value:
        await _clear(value)
        yield value
        await value.rollback()
        await _clear(value)
    await engine.dispose()


async def _clear(session: AsyncSession) -> None:
    run_ids = select(PaperTradingRunRecord.id).where(
        PaperTradingRunRecord.account_currency == "EVAL"
    )
    await session.execute(
        delete(PaperDecisionCycleRecord).where(
            PaperDecisionCycleRecord.paper_run_id.in_(run_ids)
        )
    )
    await session.execute(
        delete(PaperTradeRecord).where(PaperTradeRecord.run_id.in_(run_ids))
    )
    await session.execute(
        delete(PaperPositionSnapshotRecord).where(
            PaperPositionSnapshotRecord.portfolio_snapshot_id.in_(
                select(PaperPortfolioSnapshotRecord.id).where(
                    PaperPortfolioSnapshotRecord.run_id.in_(run_ids)
                )
            )
        )
    )
    await session.execute(
        delete(PaperPortfolioSnapshotRecord).where(
            PaperPortfolioSnapshotRecord.run_id.in_(run_ids)
        )
    )
    await session.execute(
        delete(PaperPositionRecord).where(PaperPositionRecord.run_id.in_(run_ids))
    )
    await session.execute(
        delete(DecisionSnapshotRecord).where(
            DecisionSnapshotRecord.source_score_id.in_(
                select(MarketScoreRecord.id).where(
                    MarketScoreRecord.calculated_at == MARKER
                )
            )
        )
    )
    await session.execute(
        delete(MarketScoreRecord).where(MarketScoreRecord.calculated_at == MARKER)
    )
    await session.execute(
        delete(MarketCandleRecord).where(MarketCandleRecord.ingested_at == MARKER)
    )
    await session.execute(
        delete(PaperTradingRunRecord).where(
            PaperTradingRunRecord.account_currency == "EVAL"
        )
    )
    await session.commit()


async def test_persisted_query_layer_uses_one_complete_authoritative_boundary(
    session: AsyncSession,
) -> None:
    run = _run()
    session.add(run)
    await session.flush()
    before_btc = _evidence(
        run.id, BTC, BOUNDARY - timedelta(minutes=10), "open_position"
    )
    before_eth = _evidence(
        run.id, ETH, BOUNDARY - timedelta(minutes=9), "open_position"
    )
    after_btc = _evidence(run.id, BTC, BOUNDARY + timedelta(minutes=1), "open_position")
    await _persist_evidence(session, before_btc, before_eth, after_btc)
    session.add_all(
        [
            _snapshot(run.id, BOUNDARY, Decimal("102.1234567890123456789")),
            _candle("BTC/USDT", "23:50:00", "23:54:59.999000", "9"),
            _candle("BTC/USDT", "23:55:00", "23:59:59.999000", "10"),
            _candle("BTC/USDT", "00:00:00", "00:04:59.999000", "11"),
            _candle("BTC/USDT", "00:25:00", "00:29:59.999000", "12"),
            _candle("BTC/USDT", "00:30:00", "00:34:59.999000", "99"),
            _candle("ETH/USDT", "00:00:00", "00:04:59.999000", "20"),
            _candle("ETH/USDT", "00:25:00", "00:29:59.999000", "22"),
        ]
    )
    await session.commit()
    persisted_run = await get_run(session, run_id=run.id)
    assert persisted_run is not None
    before_count = await _row_count(session, run.id)

    decisions = await evaluation_decisions(
        session, run_id=run.id, evaluated_until=BOUNDARY
    )
    trades = await evaluation_trades(session, run_id=run.id, evaluated_until=BOUNDARY)
    closed = await evaluation_closed_positions(
        session, run_id=run.id, evaluated_until=BOUNDARY
    )
    result = await evaluate_paper_run(session=session, run=persisted_run)

    assert {item.id for item in decisions} == {
        before_btc.decision.id,
        before_eth.decision.id,
    }
    assert {item.id for item in trades} == {before_btc.trade.id, before_eth.trade.id}
    assert {item.id for item in closed} == {
        before_btc.position.id,
        before_eth.position.id,
    }
    run_result = cast(dict[str, object], result["run"])
    strategy = cast(dict[str, object], result["strategy"])
    benchmark = cast(dict[str, object], result["benchmark"])
    comparison = cast(dict[str, object], result["comparison"])
    assert run_result["evaluated_until"] == BOUNDARY
    assert strategy["ending_equity"] == Decimal("102.123456789012345679")
    assert strategy["total_return"] == Decimal("0.021234567890123457")
    assert strategy["current_drawdown"] == Decimal(0).quantize(Decimal("1e-18"))
    assert strategy["decision_count"] == strategy["trade_count"] == 2
    assert strategy["closed_position_count"] == 2
    assert benchmark["status"] == comparison["status"] == "available"
    assert benchmark["instrument_count"] == 2
    assert all(
        item["allocation"] == Decimal("50.000000000000000000")
        for item in cast(list[dict[str, object]], benchmark["instruments"])
    )
    assert await _row_count(session, run.id) == before_count

    session.add(_snapshot(run.id, BOUNDARY + timedelta(minutes=30), Decimal("999")))
    await session.commit()
    bounded_points = await get_performance_points(
        session, run_id=run.id, evaluated_until=BOUNDARY
    )
    assert [point.equity for point in bounded_points] == [
        Decimal("102.123456789012345679")
    ]


async def test_benchmark_candles_use_close_boundaries_and_complete_identity(
    session: AsyncSession,
) -> None:
    run = _run()
    session.add(run)
    await session.flush()
    session.add_all(
        [
            _candle("BTC/USDT", "23:50:00", "23:54:59.999000", "9"),
            _candle("BTC/USDT", "23:55:00", "23:59:59.999000", "10"),
            _candle("BTC/USDT", "00:00:00", "00:04:59.999000", "11"),
            _candle("BTC/USDT", "00:25:00", "00:29:59.999000", "12"),
            _candle("BTC/USDT", "00:30:00", "00:34:59.999000", "99"),
        ]
    )
    await session.commit()

    entry, ending = await benchmark_candles(
        session,
        identity=BTC,
        started_at=START,
        evaluated_until=BOUNDARY,
    )
    assert entry is not None and entry.open_time == datetime(
        2097, 7, 19, 23, 55, tzinfo=UTC
    )
    assert ending is not None and ending.open_time == datetime(
        2097, 7, 20, 0, 25, tzinfo=UTC
    )
    assert entry.close_price == Decimal("10.000000000000000000")
    assert ending.close_price == Decimal("12.000000000000000000")
    for identity in (
        MarketIdentity("BTC/USDT", "other", "spot", "5m"),
        MarketIdentity("BTC/USDT", "binance", "spot", "15m"),
    ):
        assert await benchmark_candles(
            session,
            identity=identity,
            started_at=START,
            evaluated_until=BOUNDARY,
        ) == (None, None)


async def test_incomplete_persisted_benchmark_evidence_does_not_reweight(
    session: AsyncSession,
) -> None:
    run = _run()
    session.add(run)
    await session.flush()
    btc = _evidence(run.id, BTC, BOUNDARY - timedelta(minutes=10), "open_position")
    eth = _evidence(run.id, ETH, BOUNDARY - timedelta(minutes=9), "open_position")
    await _persist_evidence(session, btc, eth)
    session.add_all(
        [
            _snapshot(run.id, BOUNDARY, Decimal("100")),
            _candle("BTC/USDT", "00:00:00", "00:04:59.999000", "10"),
            _candle("BTC/USDT", "00:25:00", "00:29:59.999000", "12"),
        ]
    )
    await session.commit()
    persisted_run = await get_run(session, run_id=run.id)
    assert persisted_run is not None

    result = await evaluate_paper_run(session=session, run=persisted_run)
    benchmark = cast(dict[str, object], result["benchmark"])
    instruments = cast(list[dict[str, object]], benchmark["instruments"])
    assert benchmark["status"] == "unavailable"
    assert benchmark["unavailable_reason"] == "incomplete_candle_evidence"
    assert benchmark["ending_equity"] is None
    assert len(instruments) == 2
    assert [item["allocation"] for item in instruments] == [
        Decimal("50.000000000000000000"),
        Decimal("50.000000000000000000"),
    ]
    assert [item["status"] for item in instruments] == ["available", "unavailable"]


class PersistedEvidence:
    def __init__(
        self,
        score: MarketScoreRecord,
        decision: DecisionSnapshotRecord,
        position: PaperPositionRecord,
        trade: PaperTradeRecord,
        cycle: PaperDecisionCycleRecord,
    ) -> None:
        self.score = score
        self.decision = decision
        self.position = position
        self.trade = trade
        self.cycle = cycle


def _evidence(
    run_id: UUID, identity: MarketIdentity, timestamp: datetime, action: str
) -> PersistedEvidence:
    score = _score(identity, timestamp)
    decision = _decision(identity, timestamp, score.id, action)
    position = _position(run_id, identity, timestamp)
    trade = _trade(run_id, identity, timestamp, decision.id, position.id)
    cycle = _cycle(run_id, identity, timestamp, score.id, decision.id, trade.id, action)
    return PersistedEvidence(score, decision, position, trade, cycle)


async def _persist_evidence(session: AsyncSession, *values: PersistedEvidence) -> None:
    session.add_all([value.score for value in values])
    await session.flush()
    session.add_all(
        [item for value in values for item in (value.decision, value.position)]
    )
    await session.flush()
    session.add_all([value.trade for value in values])
    await session.flush()
    session.add_all([value.cycle for value in values])
    await session.flush()


def _identity_fields(identity: MarketIdentity) -> dict[str, str]:
    return {
        "asset": identity.asset,
        "provider": identity.provider,
        "market_type": identity.market_type,
        "interval": identity.interval,
    }


def _run() -> PaperTradingRunRecord:
    return PaperTradingRunRecord(
        id=uuid4(),
        status="stopped",
        account_currency="EVAL",
        initial_cash=Decimal("100"),
        current_cash=Decimal("102"),
        configuration_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=START,
        stopped_at=BOUNDARY + timedelta(hours=1),
        stop_reason="evaluation integration test",
        predecessor_run_id=None,
    )


def _score(identity: MarketIdentity, timestamp: datetime) -> MarketScoreRecord:
    return MarketScoreRecord(
        id=uuid4(),
        asset=identity.asset,
        provider=identity.provider,
        market_type=identity.market_type,
        interval=identity.interval,
        candle_open_time=timestamp - timedelta(minutes=5),
        model_name="baseline_technical",
        model_version=1,
        calculated_at=MARKER,
        trend_score=Decimal("0"),
        momentum_score=Decimal("0"),
        volume_activity=Decimal("0"),
        volatility_ratio=Decimal("0"),
        directional_score=Decimal("0"),
        volume_confirmation_factor=Decimal("1"),
        overall_score=Decimal("0"),
        signal_strength=Decimal("0"),
        trend_contribution=Decimal("0"),
        rsi_contribution=Decimal("0"),
        return_1_contribution=Decimal("0"),
        return_12_contribution=Decimal("0"),
        interpretation="neutral",
    )


def _decision(
    identity: MarketIdentity,
    timestamp: datetime,
    score_id: UUID,
    action: str,
) -> DecisionSnapshotRecord:
    target = Decimal("15") if action == "open_position" else None
    position = Decimal("100") if action == "sell_all" else None
    return DecisionSnapshotRecord(
        id=uuid4(),
        **_identity_fields(identity),
        decision_timestamp=timestamp,
        source_candle_open_time=timestamp - timedelta(minutes=5),
        source_market_price=Decimal("10"),
        source_score_id=score_id,
        source_score_model_name="baseline_technical",
        source_score_model_version=1,
        source_overall_score=Decimal("0"),
        strategy_name="baseline_long_only",
        strategy_version=1,
        configuration_version=1,
        action=action,
        requested_target_portfolio_percentage=target,
        requested_position_percentage=position,
        applied_rule=(
            "exit_threshold_breached" if action == "sell_all" else "entry_threshold_met"
        ),
        explanation="Persisted integration evidence.",
        position_is_open=action == "sell_all",
    )


def _position(
    run_id: UUID, identity: MarketIdentity, timestamp: datetime
) -> PaperPositionRecord:
    return PaperPositionRecord(
        id=uuid4(),
        run_id=run_id,
        **_identity_fields(identity),
        status="closed",
        quantity=Decimal("0"),
        average_entry_price=Decimal("10"),
        total_entry_notional=Decimal("10"),
        accumulated_entry_fees=Decimal("0.01"),
        opened_at=timestamp - timedelta(minutes=2),
        closed_at=timestamp,
        exit_price=Decimal("11"),
        gross_realized_pnl=Decimal("1"),
        net_realized_pnl=Decimal("0.99"),
    )


def _trade(
    run_id: UUID,
    identity: MarketIdentity,
    timestamp: datetime,
    decision_id: UUID,
    position_id: UUID,
) -> PaperTradeRecord:
    return PaperTradeRecord(
        id=uuid4(),
        run_id=run_id,
        decision_snapshot_id=decision_id,
        position_id=position_id,
        side="buy",
        **_identity_fields(identity),
        observed_market_price=Decimal("10"),
        slippage_rate=Decimal("0.0005"),
        execution_price=Decimal("10.005"),
        quantity=Decimal("1"),
        executed_notional=Decimal("10.005"),
        fee_rate=Decimal("0.001"),
        fee_amount=Decimal("0.010005"),
        cash_impact=Decimal("-10.015005"),
        executed_at=timestamp,
        configuration_version=1,
        equity_before=Decimal("100"),
    )


def _cycle(
    run_id: UUID,
    identity: MarketIdentity,
    timestamp: datetime,
    score_id: UUID,
    decision_id: UUID,
    trade_id: UUID,
    action: str,
) -> PaperDecisionCycleRecord:
    result = action if action in {"open_position", "sell_all"} else "no_action"
    return PaperDecisionCycleRecord(
        id=uuid4(),
        paper_run_id=run_id,
        **_identity_fields(identity),
        strategy_name="baseline_long_only",
        strategy_version=1,
        configuration_version=1,
        scheduled_boundary=timestamp - timedelta(minutes=1),
        source_score_id=score_id,
        decision_snapshot_id=decision_id,
        paper_trade_id=trade_id if result != "no_action" else None,
        result=result,
        started_at=timestamp - timedelta(seconds=30),
        completed_at=timestamp,
        status="completed",
    )


def _snapshot(
    run_id: UUID, boundary: datetime, equity: Decimal
) -> PaperPortfolioSnapshotRecord:
    return PaperPortfolioSnapshotRecord(
        id=uuid4(),
        run_id=run_id,
        scheduled_boundary=boundary,
        recorded_at=boundary + timedelta(minutes=2),
        account_currency="EVAL",
        initial_cash=Decimal("100"),
        current_cash=equity,
        positions_market_value=Decimal("0"),
        equity=equity,
        total_pnl=equity - Decimal("100"),
        total_return=(equity - Decimal("100")) / Decimal("100"),
        configuration_version=1,
    )


def _candle(
    asset: str, open_clock: str, close_clock: str, price: str
) -> MarketCandleRecord:
    open_day = 19 if open_clock.startswith("23:") else 20
    close_day = 19 if close_clock.startswith("23:") else 20
    open_time = datetime.fromisoformat(f"2097-07-{open_day:02d}T{open_clock}+00:00")
    close_time = datetime.fromisoformat(f"2097-07-{close_day:02d}T{close_clock}+00:00")
    value = Decimal(price)
    return MarketCandleRecord(
        id=uuid4(),
        asset=asset,
        provider="binance",
        market_type="spot",
        interval="5m",
        open_time=open_time,
        close_time=close_time,
        open_price=value,
        high_price=value,
        low_price=value,
        close_price=value,
        base_asset_volume=Decimal("1"),
        quote_asset_volume=value,
        trade_count=1,
        taker_buy_base_asset_volume=Decimal("0.5"),
        taker_buy_quote_asset_volume=value / Decimal("2"),
        ingested_at=MARKER,
    )


async def _row_count(session: AsyncSession, run_id: UUID) -> int:
    counts = []
    for record in (
        PaperPortfolioSnapshotRecord,
        PaperPositionRecord,
        PaperTradeRecord,
        PaperDecisionCycleRecord,
    ):
        key = record.run_id if hasattr(record, "run_id") else record.paper_run_id
        counts.append(
            int(
                await session.scalar(
                    select(func.count()).select_from(record).where(key == run_id)
                )
                or 0
            )
        )
    return sum(counts)
