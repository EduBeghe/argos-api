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
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from app.paper.equity_snapshots import (
    PaperPortfolioSnapshotRecord,
    PaperPositionSnapshotRecord,
    get_equity_snapshot,
    list_equity_snapshots,
    persist_equity_snapshot,
)
from app.paper.models import PaperPositionStatus
from app.paper.persistence import PaperPositionRecord, PaperTradingRunRecord, get_run
from app.paper.portfolio import PaperPortfolioValuationError
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


def _candle(*, open_time: datetime, price: Decimal) -> MarketCandleRecord:
    return to_record(
        make_candle(
            open_time=open_time,
            close_time=open_time + timedelta(minutes=5),
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
            ingested_at=MARKER,
        )
    )


async def test_snapshot_parent_children_are_atomic_historical_and_idempotent(
    session: AsyncSession,
) -> None:
    run = await _run_with_position(session)
    boundary = NOW + timedelta(minutes=30)
    session.add_all(
        [
            _candle(open_time=NOW, price=Decimal("110")),
            _candle(open_time=NOW + timedelta(minutes=5), price=Decimal("120")),
            _candle(open_time=boundary, price=Decimal("999")),
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
    assert len(first.snapshot.positions) == 1
    position = first.snapshot.positions[0]
    assert position.mark_price == Decimal("120.000000000000000000")
    assert position.mark_candle_open_time == NOW + timedelta(minutes=5)
    assert position.cost_basis == Decimal("201.000000000000000000")
    assert (
        len((await session.scalars(select(PaperPortfolioSnapshotRecord.id))).all()) == 1
    )
    assert (
        len((await session.scalars(select(PaperPositionSnapshotRecord.id))).all()) == 1
    )
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
