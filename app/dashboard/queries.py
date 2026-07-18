from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.models import DecisionSnapshot
from app.decisions.persistence import DecisionSnapshotRecord
from app.market.catalog import Instrument
from app.paper.models import PaperPositionStatus
from app.paper.persistence import PaperPositionRecord
from app.scores.models import MarketScore
from app.scores.persistence import MarketScoreRecord


def _identities(instruments: Sequence[Instrument]) -> list[tuple[str, str, str, str]]:
    return [
        (
            item.asset.value,
            item.provider.value,
            item.market_type.value,
            item.interval.value,
        )
        for item in instruments
    ]


async def get_latest_scores_batch(
    session: AsyncSession, *, instruments: Sequence[Instrument]
) -> tuple[MarketScore, ...]:
    if not instruments:
        return ()
    records = (
        await session.scalars(
            select(MarketScoreRecord)
            .where(
                tuple_(
                    MarketScoreRecord.asset,
                    MarketScoreRecord.provider,
                    MarketScoreRecord.market_type,
                    MarketScoreRecord.interval,
                ).in_(_identities(instruments))
            )
            .distinct(
                MarketScoreRecord.asset,
                MarketScoreRecord.provider,
                MarketScoreRecord.market_type,
                MarketScoreRecord.interval,
            )
            .order_by(
                MarketScoreRecord.asset,
                MarketScoreRecord.provider,
                MarketScoreRecord.market_type,
                MarketScoreRecord.interval,
                MarketScoreRecord.candle_open_time.desc(),
                MarketScoreRecord.id.desc(),
            )
        )
    ).all()
    return tuple(
        MarketScore.model_validate(
            {
                column.name: getattr(record, column.name)
                for column in record.__table__.columns
            }
        )
        for record in records
    )


async def get_latest_decisions_batch(
    session: AsyncSession, *, instruments: Sequence[Instrument]
) -> tuple[DecisionSnapshot, ...]:
    if not instruments:
        return ()
    records = (
        await session.scalars(
            select(DecisionSnapshotRecord)
            .where(
                tuple_(
                    DecisionSnapshotRecord.asset,
                    DecisionSnapshotRecord.provider,
                    DecisionSnapshotRecord.market_type,
                    DecisionSnapshotRecord.interval,
                ).in_(_identities(instruments))
            )
            .distinct(
                DecisionSnapshotRecord.asset,
                DecisionSnapshotRecord.provider,
                DecisionSnapshotRecord.market_type,
                DecisionSnapshotRecord.interval,
            )
            .order_by(
                DecisionSnapshotRecord.asset,
                DecisionSnapshotRecord.provider,
                DecisionSnapshotRecord.market_type,
                DecisionSnapshotRecord.interval,
                DecisionSnapshotRecord.decision_timestamp.desc(),
                DecisionSnapshotRecord.id.desc(),
            )
        )
    ).all()
    return tuple(
        DecisionSnapshot.model_validate(
            {
                column.name: getattr(record, column.name)
                for column in record.__table__.columns
            }
        )
        for record in records
    )


async def get_open_positions_batch(
    session: AsyncSession, *, run_id: UUID, instruments: Sequence[Instrument]
) -> tuple[PaperPositionRecord, ...]:
    if not instruments:
        return ()
    return tuple(
        (
            await session.scalars(
                select(PaperPositionRecord).where(
                    PaperPositionRecord.run_id == run_id,
                    PaperPositionRecord.status == PaperPositionStatus.OPEN.value,
                    tuple_(
                        PaperPositionRecord.asset,
                        PaperPositionRecord.provider,
                        PaperPositionRecord.market_type,
                        PaperPositionRecord.interval,
                    ).in_(_identities(instruments)),
                )
            )
        ).all()
    )

