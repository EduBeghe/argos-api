import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from app.paper.equity_snapshots import (
    PaperPortfolioSnapshotRecord,
    PaperPositionSnapshotRecord,
    PreparedPortfolioSnapshot,
    get_equity_snapshot,
    list_equity_snapshots,
    persist_equity_snapshot,
    prepare_snapshot_values,
)
from app.paper.models import PaperPositionStatus
from app.paper.performance import calculate_performance, get_performance_points
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradingRunRecord,
    get_run,
    run_domain,
)
from app.paper.portfolio import PaperPortfolioValuation, PaperPortfolioValuationError
from app.paper.reset import reset_paper_account
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
NOW = datetime(2026, 7, 18, 12, tzinfo=UTC)
MARKER = datetime(2098, 1, 1, tzinfo=UTC)


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
    for record in (
        PaperPositionSnapshotRecord,
        PaperPortfolioSnapshotRecord,
        PaperPositionRecord,
        PaperTradingRunRecord,
    ):
        await session.execute(delete(record))
    await session.execute(
        delete(MarketCandleRecord).where(MarketCandleRecord.ingested_at == MARKER)
    )
    await session.commit()


async def _run_with_position(session: AsyncSession) -> PaperTradingRunRecord:
    run = await reset_paper_account(session=session, started_at=NOW)
    session.add(
        PaperPositionRecord(
            id=uuid4(),
            run_id=run.id,
            asset=Asset.BTC_USDT.value,
            provider=Provider.BINANCE.value,
            market_type=MarketType.SPOT.value,
            interval=Interval.FIVE_MINUTES.value,
            status=PaperPositionStatus.OPEN.value,
            quantity=Decimal("2"),
            average_entry_price=Decimal("100"),
            total_entry_notional=Decimal("200"),
            accumulated_entry_fees=Decimal("1"),
            opened_at=NOW,
            closed_at=None,
            exit_price=None,
            gross_realized_pnl=None,
            net_realized_pnl=None,
        )
    )
    await session.flush()
    record = await get_run(session, run_id=run.id)
    assert record is not None
    return record


def _candle(
    *,
    open_time: datetime,
    price: Decimal,
    asset: Asset = Asset.BTC_USDT,
    provider: Provider = Provider.BINANCE,
    market_type: MarketType = MarketType.SPOT,
    interval: Interval = Interval.FIVE_MINUTES,
) -> MarketCandleRecord:
    return to_record(
        make_candle(
            open_time=open_time,
            close_time=open_time + timedelta(minutes=5),
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
            ingested_at=MARKER,
        )
    )


async def test_snapshot_parent_children_are_atomic_historical_and_idempotent(
    session: AsyncSession,
) -> None:
    run = await _run_with_position(session)
    boundary = NOW + timedelta(minutes=30)
    session.add(
        PaperPositionRecord(
            id=uuid4(),
            run_id=run.id,
            asset=Asset.ETH_USDT.value,
            provider=Provider.BINANCE.value,
            market_type=MarketType.SPOT.value,
            interval=Interval.FIVE_MINUTES.value,
            status=PaperPositionStatus.OPEN.value,
            quantity=Decimal("1.0000000000000000006"),
            average_entry_price=Decimal("10.0000000000000000004"),
            total_entry_notional=Decimal("10.0000000000000000004"),
            accumulated_entry_fees=Decimal("0.0000000000000000006"),
            opened_at=NOW,
            closed_at=None,
            exit_price=None,
            gross_realized_pnl=None,
            net_realized_pnl=None,
        )
    )
    session.add_all(
        [
            _candle(open_time=NOW, price=Decimal("110")),
            _candle(open_time=NOW + timedelta(minutes=5), price=Decimal("120")),
            _candle(open_time=boundary, price=Decimal("999")),
            _candle(
                open_time=NOW + timedelta(minutes=5),
                price=Decimal("20.0000000000000000006"),
                asset=Asset.ETH_USDT,
            ),
        ]
    )
    first = await persist_equity_snapshot(
        session=session,
        run=run,
        scheduled_boundary=boundary,
        recorded_at=boundary + timedelta(minutes=2),
    )
    await session.commit()
    repeated = await persist_equity_snapshot(
        session=session,
        run=run,
        scheduled_boundary=boundary,
        recorded_at=boundary + timedelta(minutes=3),
    )
    assert first.created is True and repeated.created is False
    assert repeated.snapshot.id == first.snapshot.id
    assert len(first.snapshot.positions) == 2
    position = next(
        item for item in first.snapshot.positions if item.asset == Asset.BTC_USDT.value
    )
    assert position.mark_price == Decimal("120.000000000000000000")
    assert position.mark_candle_open_time == NOW + timedelta(minutes=5)
    assert position.cost_basis == Decimal("201.000000000000000000")
    assert (
        len((await session.scalars(select(PaperPortfolioSnapshotRecord.id))).all()) == 1
    )
    assert (
        len((await session.scalars(select(PaperPositionSnapshotRecord.id))).all()) == 2
    )
    session.expunge_all()
    reread = await get_equity_snapshot(session, first.snapshot.id)
    assert reread is not None
    assert reread.positions_market_value == sum(
        (item.market_value for item in reread.positions), start=Decimal(0)
    )
    assert reread.equity == reread.current_cash + reread.positions_market_value
    assert reread.total_pnl == reread.equity - reread.initial_cash
    session.add(
        PaperPortfolioSnapshotRecord(
            id=uuid4(),
            run_id=run.id,
            scheduled_boundary=boundary,
            recorded_at=boundary + timedelta(minutes=4),
            account_currency="USDT",
            initial_cash=Decimal("10000"),
            current_cash=Decimal("10000"),
            positions_market_value=Decimal("0"),
            equity=Decimal("10000"),
            total_pnl=Decimal("0"),
            total_return=Decimal("0"),
            configuration_version=1,
        )
    )
    with pytest.raises(IntegrityError):
        await session.flush()
    await session.rollback()


async def test_empty_portfolio_persists_parent_without_children(
    session: AsyncSession,
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    record = await get_run(session, run_id=run.id)
    assert record is not None
    result = await persist_equity_snapshot(
        session=session,
        run=record,
        scheduled_boundary=NOW + timedelta(minutes=30),
        recorded_at=NOW + timedelta(minutes=32),
    )
    assert result.created is True
    assert result.snapshot.positions_market_value == 0
    assert result.snapshot.positions == []


async def test_missing_compatible_candle_persists_nothing(
    session: AsyncSession,
) -> None:
    run = await _run_with_position(session)
    with pytest.raises(PaperPortfolioValuationError):
        await persist_equity_snapshot(
            session=session,
            run=run,
            scheduled_boundary=NOW + timedelta(minutes=30),
            recorded_at=NOW + timedelta(minutes=32),
        )
    await session.rollback()
    assert await session.scalar(select(PaperPortfolioSnapshotRecord.id)) is None


async def test_child_insert_failure_rolls_back_parent(session: AsyncSession) -> None:
    run = await _run_with_position(session)
    boundary = NOW + timedelta(minutes=30)
    session.add(_candle(open_time=NOW, price=Decimal("120")))

    def invalid_child(
        valuation: PaperPortfolioValuation,
    ) -> PreparedPortfolioSnapshot:
        prepared = prepare_snapshot_values(valuation)
        child = replace(prepared.positions[0], asset="")
        return replace(prepared, positions=(child,))

    with (
        patch(
            "app.paper.equity_snapshots.prepare_snapshot_values",
            side_effect=invalid_child,
        ),
        pytest.raises(IntegrityError),
    ):
        await persist_equity_snapshot(
            session=session,
            run=run,
            scheduled_boundary=boundary,
            recorded_at=boundary + timedelta(minutes=2),
        )
    await session.rollback()
    assert await session.scalar(select(PaperPortfolioSnapshotRecord.id)) is None
    assert await session.scalar(select(PaperPositionSnapshotRecord.id)) is None


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("provider", "different-provider"),
        ("market_type", "different-market"),
        ("interval", "different"),
    ],
)
async def test_mark_selection_does_not_mix_instrument_identity_fields(
    session: AsyncSession, attribute: str, value: str
) -> None:
    run = await _run_with_position(session)
    position = await session.scalar(select(PaperPositionRecord))
    assert position is not None
    setattr(position, attribute, value)
    session.add(_candle(open_time=NOW, price=Decimal("120")))
    await session.flush()
    with pytest.raises(PaperPortfolioValuationError):
        await persist_equity_snapshot(
            session=session,
            run=run,
            scheduled_boundary=NOW + timedelta(minutes=30),
            recorded_at=NOW + timedelta(minutes=32),
        )
    await session.rollback()
    assert await session.scalar(select(PaperPortfolioSnapshotRecord.id)) is None


async def test_snapshot_queries_are_ordered_paginated_and_load_positions(
    session: AsyncSession,
) -> None:
    run = await reset_paper_account(session=session, started_at=NOW)
    record = await get_run(session, run_id=run.id)
    assert record is not None
    ids = []
    for minutes in (30, 60):
        result = await persist_equity_snapshot(
            session=session,
            run=record,
            scheduled_boundary=NOW + timedelta(minutes=minutes),
            recorded_at=NOW + timedelta(minutes=minutes + 2),
        )
        ids.append(result.snapshot.id)
    await session.commit()
    page = await list_equity_snapshots(
        session=session, run_id=run.id, limit=1, offset=0, order="desc"
    )
    historical = await get_equity_snapshot(session, ids[0])
    assert [item.id for item in page] == [ids[1]]
    assert historical is not None and historical.positions == []


async def test_concurrent_persistence_creates_one_complete_snapshot(
    session: AsyncSession,
) -> None:
    run = await _run_with_position(session)
    boundary = NOW + timedelta(minutes=30)
    session.add(_candle(open_time=NOW, price=Decimal("120")))
    await session.commit()
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def attempt() -> tuple[bool, int]:
        async with factory() as concurrent_session:
            persisted_run = await get_run(concurrent_session, run_id=run.id)
            assert persisted_run is not None
            result = await persist_equity_snapshot(
                session=concurrent_session,
                run=persisted_run,
                scheduled_boundary=boundary,
                recorded_at=boundary + timedelta(minutes=2),
            )
            await concurrent_session.commit()
            return result.created, len(result.snapshot.positions)

    outcomes = await asyncio.gather(attempt(), attempt())
    assert sorted(created for created, _ in outcomes) == [False, True]
    assert [positions_count for _, positions_count in outcomes] == [1, 1]
    async with factory() as verification:
        parents = (
            await verification.scalars(select(PaperPortfolioSnapshotRecord.id))
        ).all()
        children = (
            await verification.scalars(select(PaperPositionSnapshotRecord.id))
        ).all()
        assert len(parents) == len(children) == 1
        complete = await get_equity_snapshot(verification, parents[0])
        assert complete is not None and len(complete.positions) == 1
    await engine.dispose()


async def test_performance_reads_complete_ordered_run_history_without_positions(
    session: AsyncSession,
) -> None:
    active = await reset_paper_account(session=session, started_at=NOW)
    active_record = await get_run(session, run_id=active.id)
    assert active_record is not None
    stopped = PaperTradingRunRecord(
        id=uuid4(),
        status="stopped",
        account_currency="EUR",
        initial_cash=Decimal("5000"),
        current_cash=Decimal("5100"),
        configuration_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=NOW - timedelta(days=1),
        stopped_at=NOW,
        stop_reason="integration test",
        predecessor_run_id=None,
    )
    session.add(stopped)
    await session.flush()
    for index in range(501):
        boundary = NOW + timedelta(minutes=30 * (index + 1))
        session.add(
            PaperPortfolioSnapshotRecord(
                id=uuid4(),
                run_id=active.id,
                scheduled_boundary=boundary,
                recorded_at=boundary + timedelta(minutes=2),
                account_currency="USDT",
                initial_cash=Decimal("10000"),
                current_cash=Decimal("10000"),
                positions_market_value=Decimal(index),
                equity=Decimal(10000 + index),
                total_pnl=Decimal(index),
                total_return=Decimal(index) / Decimal(10000),
                configuration_version=1,
            )
        )
    other_boundary = NOW + timedelta(minutes=15)
    session.add(
        PaperPortfolioSnapshotRecord(
            id=uuid4(),
            run_id=stopped.id,
            scheduled_boundary=other_boundary,
            recorded_at=other_boundary + timedelta(minutes=2),
            account_currency="EUR",
            initial_cash=Decimal("5000"),
            current_cash=Decimal("5100"),
            positions_market_value=Decimal(0),
            equity=Decimal("5100"),
            total_pnl=Decimal("100"),
            total_return=Decimal("0.02"),
            configuration_version=1,
        )
    )
    await session.commit()

    active_points = await get_performance_points(session, run_id=active.id)
    stopped_points = await get_performance_points(session, run_id=stopped.id)
    active_performance = calculate_performance(
        run=run_domain(active_record), snapshots=active_points
    )
    stopped_performance = calculate_performance(
        run=run_domain(stopped), snapshots=stopped_points
    )

    assert len(active_points) == 501
    assert [point.scheduled_boundary for point in active_points] == sorted(
        point.scheduled_boundary for point in active_points
    )
    assert active_performance.latest_equity == Decimal("10500.000000000000000000")
    assert active_performance.latest_snapshot_boundary == NOW + timedelta(
        minutes=30 * 501
    )
    assert stopped_performance.snapshot_count == 1
    assert stopped_performance.latest_equity == Decimal("5100.000000000000000000")
    assert all(point.equity != Decimal("5100") for point in active_points)
    assert active_record.current_cash == Decimal("10000")
