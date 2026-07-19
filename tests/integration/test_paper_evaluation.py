from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.paper.equity_snapshots import PaperPortfolioSnapshotRecord
from app.paper.evaluation_service import evaluate_paper_run
from app.paper.persistence import (
    PaperPositionRecord,
    PaperTradingRunRecord,
    get_run,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2097, 7, 19, 12, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as value:
        yield value
        await value.rollback()
        run_ids = select(PaperTradingRunRecord.id).where(
            PaperTradingRunRecord.account_currency == "EVAL"
        )
        await value.execute(
            delete(PaperPositionRecord).where(PaperPositionRecord.run_id.in_(run_ids))
        )
        await value.execute(
            delete(PaperPortfolioSnapshotRecord).where(
                PaperPortfolioSnapshotRecord.run_id.in_(run_ids)
            )
        )
        await value.execute(
            delete(PaperTradingRunRecord).where(
                PaperTradingRunRecord.account_currency == "EVAL"
            )
        )
        await value.commit()
    await engine.dispose()


async def test_evaluation_uses_latest_snapshot_boundary_and_performs_no_writes(
    session: AsyncSession,
) -> None:
    run = PaperTradingRunRecord(
        id=uuid4(),
        status="stopped",
        account_currency="EVAL",
        initial_cash=Decimal("100.0000000000000000004"),
        current_cash=Decimal("102"),
        configuration_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=START,
        stopped_at=START + timedelta(hours=2),
        stop_reason="evaluation integration test",
        predecessor_run_id=None,
    )
    boundary = START + timedelta(minutes=30)
    snapshot = PaperPortfolioSnapshotRecord(
        id=uuid4(),
        run_id=run.id,
        scheduled_boundary=boundary,
        recorded_at=boundary + timedelta(minutes=2),
        account_currency="EVAL",
        initial_cash=Decimal("100.0000000000000000004"),
        current_cash=Decimal("102.1234567890123456789"),
        positions_market_value=Decimal("0"),
        equity=Decimal("102.1234567890123456789"),
        total_pnl=Decimal("2.1234567890123456785"),
        total_return=Decimal("0.021234567890123457"),
        configuration_version=1,
    )
    before = _closed_position(run.id, boundary - timedelta(minutes=1), Decimal("5"))
    after = _closed_position(run.id, boundary + timedelta(minutes=1), Decimal("99"))
    session.add(run)
    await session.flush()
    session.add_all([snapshot, before, after])
    await session.commit()
    persisted = await get_run(session, run_id=run.id)
    assert persisted is not None
    before_count = await _row_count(session, run.id)

    result = await evaluate_paper_run(session=session, run=persisted)

    after_count = await _row_count(session, run.id)
    assert before_count == after_count
    assert result["run"]["latest_snapshot_id"] == snapshot.id  # type: ignore[index]
    assert result["run"]["evaluated_until"] == boundary  # type: ignore[index]
    strategy = result["strategy"]
    assert strategy["ending_equity"] == Decimal("102.123456789012345679")  # type: ignore[index]
    assert strategy["closed_position_count"] == 1  # type: ignore[index]
    assert strategy["realized_net_pnl"] == Decimal("5.000000000000000000")  # type: ignore[index]
    assert result["benchmark"]["status"] == "unavailable"  # type: ignore[index]
    assert result["instruments"][0]["asset"] == "BTC/USDT"  # type: ignore[index]


def _closed_position(
    run_id: UUID, closed_at: datetime, pnl: Decimal
) -> PaperPositionRecord:
    return PaperPositionRecord(
        id=uuid4(),
        run_id=run_id,
        asset="BTC/USDT",
        provider="binance",
        market_type="spot",
        interval="5m",
        status="closed",
        quantity=Decimal("0"),
        average_entry_price=Decimal("10"),
        total_entry_notional=Decimal("10"),
        accumulated_entry_fees=Decimal("0.01"),
        opened_at=START,
        closed_at=closed_at,
        exit_price=Decimal("11"),
        gross_realized_pnl=pnl + Decimal("0.01"),
        net_realized_pnl=pnl,
    )


async def _row_count(session: AsyncSession, run_id: UUID) -> int:
    counts = []
    for record in (PaperPortfolioSnapshotRecord, PaperPositionRecord):
        counts.append(
            int(
                await session.scalar(
                    select(func.count())
                    .select_from(record)
                    .where(record.run_id == run_id)
                )
                or 0
            )
        )
    return sum(counts)
