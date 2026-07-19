from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.paper.config import quantize_financial
from app.paper.equity_snapshots import PaperPortfolioSnapshotRecord
from app.paper.models import PaperTradingRun


@dataclass(frozen=True, slots=True)
class EquityPoint:
    scheduled_boundary: datetime
    equity: Decimal


@dataclass(frozen=True, slots=True)
class PaperRunPerformance:
    snapshot_count: int
    initial_equity: Decimal
    latest_equity: Decimal
    absolute_pnl: Decimal
    total_return: Decimal | None
    peak_equity: Decimal
    lowest_equity: Decimal
    current_drawdown: Decimal
    max_drawdown: Decimal
    first_snapshot_boundary: datetime | None
    latest_snapshot_boundary: datetime | None
    evaluated_from: datetime
    evaluated_until: datetime


def calculate_performance(
    *, run: PaperTradingRun, snapshots: tuple[EquityPoint, ...]
) -> PaperRunPerformance:
    ordered = tuple(sorted(snapshots, key=lambda point: point.scheduled_boundary))
    initial_equity = quantize_financial(run.initial_cash)
    if initial_equity < 0:
        raise ValueError("Paper Run initial equity cannot be negative")

    equities = tuple(quantize_financial(point.equity) for point in ordered)
    if any(equity < 0 for equity in equities):
        raise ValueError("Paper equity snapshots cannot contain negative equity")

    latest_equity = equities[-1] if equities else initial_equity
    absolute_pnl = quantize_financial(latest_equity - initial_equity)
    total_return = (
        quantize_financial(absolute_pnl / initial_equity)
        if initial_equity > 0
        else None
    )
    running_peak = initial_equity
    peak_equity = initial_equity
    lowest_equity = initial_equity
    current_drawdown = Decimal(0)
    max_drawdown = Decimal(0)
    for equity in equities:
        running_peak = max(running_peak, equity)
        peak_equity = max(peak_equity, equity)
        lowest_equity = min(lowest_equity, equity)
        current_drawdown = (
            quantize_financial((running_peak - equity) / running_peak)
            if running_peak > 0
            else Decimal(0)
        )
        max_drawdown = max(max_drawdown, current_drawdown)

    first_boundary = ordered[0].scheduled_boundary if ordered else None
    latest_boundary = ordered[-1].scheduled_boundary if ordered else None
    return PaperRunPerformance(
        snapshot_count=len(ordered),
        initial_equity=initial_equity,
        latest_equity=latest_equity,
        absolute_pnl=absolute_pnl,
        total_return=total_return,
        peak_equity=peak_equity,
        lowest_equity=lowest_equity,
        current_drawdown=current_drawdown,
        max_drawdown=max_drawdown,
        first_snapshot_boundary=first_boundary,
        latest_snapshot_boundary=latest_boundary,
        evaluated_from=run.started_at,
        evaluated_until=latest_boundary or run.started_at,
    )


async def get_performance_points(
    session: AsyncSession,
    *,
    run_id: UUID,
    evaluated_until: datetime | None = None,
) -> tuple[EquityPoint, ...]:
    statement = select(
        PaperPortfolioSnapshotRecord.scheduled_boundary,
        PaperPortfolioSnapshotRecord.equity,
    ).where(PaperPortfolioSnapshotRecord.run_id == run_id)
    if evaluated_until is not None:
        statement = statement.where(
            PaperPortfolioSnapshotRecord.scheduled_boundary <= evaluated_until
        )
    rows = (
        await session.execute(
            statement.order_by(
                PaperPortfolioSnapshotRecord.scheduled_boundary.asc(),
                PaperPortfolioSnapshotRecord.id.asc(),
            )
        )
    ).all()
    return tuple(EquityPoint(boundary, equity) for boundary, equity in rows)
