from datetime import datetime
from typing import cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.decisions.persistence import DecisionSnapshotRecord
from app.market.persistence import MarketCandleRecord
from app.paper.decision_cycles import PaperDecisionCycleRecord
from app.paper.equity_snapshots import PaperPortfolioSnapshotRecord
from app.paper.evaluation import CandleEvidence, MarketIdentity
from app.paper.persistence import PaperPositionRecord, PaperTradeRecord


async def latest_snapshot_and_count(
    session: AsyncSession, *, run_id: UUID
) -> tuple[PaperPortfolioSnapshotRecord | None, int]:
    snapshot = cast(
        PaperPortfolioSnapshotRecord | None,
        await session.scalar(
            select(PaperPortfolioSnapshotRecord)
            .options(selectinload(PaperPortfolioSnapshotRecord.positions))
            .where(PaperPortfolioSnapshotRecord.run_id == run_id)
            .order_by(
                PaperPortfolioSnapshotRecord.scheduled_boundary.desc(),
                PaperPortfolioSnapshotRecord.id.desc(),
            )
            .limit(1)
        ),
    )
    count = await session.scalar(
        select(func.count(PaperPortfolioSnapshotRecord.id)).where(
            PaperPortfolioSnapshotRecord.run_id == run_id
        )
    )
    return snapshot, int(count or 0)


async def evaluation_decisions(
    session: AsyncSession, *, run_id: UUID, evaluated_until: datetime
) -> list[DecisionSnapshotRecord]:
    return list(
        (
            await session.scalars(
                select(DecisionSnapshotRecord)
                .join(
                    PaperDecisionCycleRecord,
                    PaperDecisionCycleRecord.decision_snapshot_id
                    == DecisionSnapshotRecord.id,
                )
                .where(
                    PaperDecisionCycleRecord.paper_run_id == run_id,
                    DecisionSnapshotRecord.decision_timestamp <= evaluated_until,
                )
                .order_by(
                    DecisionSnapshotRecord.asset,
                    DecisionSnapshotRecord.provider,
                    DecisionSnapshotRecord.market_type,
                    DecisionSnapshotRecord.interval,
                    DecisionSnapshotRecord.decision_timestamp,
                    DecisionSnapshotRecord.id,
                )
            )
        ).all()
    )


async def evaluation_trades(
    session: AsyncSession, *, run_id: UUID, evaluated_until: datetime
) -> list[PaperTradeRecord]:
    return list(
        (
            await session.scalars(
                select(PaperTradeRecord)
                .where(
                    PaperTradeRecord.run_id == run_id,
                    PaperTradeRecord.executed_at <= evaluated_until,
                )
                .order_by(
                    PaperTradeRecord.asset,
                    PaperTradeRecord.provider,
                    PaperTradeRecord.market_type,
                    PaperTradeRecord.interval,
                    PaperTradeRecord.executed_at,
                    PaperTradeRecord.id,
                )
            )
        ).all()
    )


async def evaluation_closed_positions(
    session: AsyncSession, *, run_id: UUID, evaluated_until: datetime
) -> list[PaperPositionRecord]:
    return list(
        (
            await session.scalars(
                select(PaperPositionRecord)
                .where(
                    PaperPositionRecord.run_id == run_id,
                    PaperPositionRecord.status == "closed",
                    PaperPositionRecord.closed_at <= evaluated_until,
                )
                .order_by(
                    PaperPositionRecord.asset,
                    PaperPositionRecord.provider,
                    PaperPositionRecord.market_type,
                    PaperPositionRecord.interval,
                    PaperPositionRecord.closed_at,
                    PaperPositionRecord.id,
                )
            )
        ).all()
    )


async def benchmark_candles(
    session: AsyncSession,
    *,
    identity: MarketIdentity,
    started_at: datetime,
    evaluated_until: datetime,
) -> tuple[CandleEvidence | None, CandleEvidence | None]:
    filters = (
        MarketCandleRecord.asset == identity.asset,
        MarketCandleRecord.provider == identity.provider,
        MarketCandleRecord.market_type == identity.market_type,
        MarketCandleRecord.interval == identity.interval,
        MarketCandleRecord.close_time >= started_at,
        MarketCandleRecord.close_time <= evaluated_until,
    )
    entry = (
        await session.execute(
            select(MarketCandleRecord.open_time, MarketCandleRecord.close_price)
            .where(*filters)
            .order_by(MarketCandleRecord.close_time.asc(), MarketCandleRecord.id.asc())
            .limit(1)
        )
    ).first()
    ending = (
        await session.execute(
            select(MarketCandleRecord.open_time, MarketCandleRecord.close_price)
            .where(*filters)
            .order_by(
                MarketCandleRecord.close_time.desc(), MarketCandleRecord.id.desc()
            )
            .limit(1)
        )
    ).first()
    return (
        CandleEvidence(entry.open_time, entry.close_price) if entry else None,
        CandleEvidence(ending.open_time, ending.close_price) if ending else None,
    )
