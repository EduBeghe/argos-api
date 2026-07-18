from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    select,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.decisions.persistence import DecisionSnapshotRecord, ScoreEvidence
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord
from app.paper.persistence import PaperTradeRecord, PaperTradingRunRecord
from app.scores.persistence import MarketScoreRecord
from app.strategies.baseline_long_only import (
    REQUIRED_SCORE_MODEL_NAME,
    REQUIRED_SCORE_MODEL_VERSION,
)

MAX_EVIDENCE_AGE = timedelta(minutes=10)


class DecisionCycleStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"


class DecisionCycleError(RuntimeError):
    """Raised when a scheduled Paper Decision Cycle cannot be completed."""


class DecisionEvidenceUnavailableError(DecisionCycleError):
    """Raised when the boundary has no eligible fresh score evidence."""


class PaperDecisionCycleRecord(Base):
    __tablename__ = "paper_decision_cycles"
    __table_args__ = (
        CheckConstraint(
            "status IN ('processing', 'completed')", name="ck_paper_cycles_status"
        ),
        CheckConstraint(
            "strategy_version > 0 AND configuration_version > 0",
            name="ck_paper_cycles_versions",
        ),
        CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0 AND "
            "length(btrim(strategy_name)) > 0",
            name="ck_paper_cycles_identity",
        ),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name="ck_paper_cycles_timestamps",
        ),
        CheckConstraint(
            "(status = 'processing' AND completed_at IS NULL AND result IS NULL) OR "
            "(status = 'completed' AND completed_at IS NOT NULL AND "
            "result IS NOT NULL AND source_score_id IS NOT NULL AND "
            "decision_snapshot_id IS NOT NULL)",
            name="ck_paper_cycles_lifecycle",
        ),
        UniqueConstraint(
            "paper_run_id",
            "asset",
            "provider",
            "market_type",
            "interval",
            "strategy_name",
            "strategy_version",
            "configuration_version",
            "scheduled_boundary",
            name="uq_paper_cycles_natural_identity",
        ),
        Index(
            "ix_paper_cycles_run_boundary", "paper_run_id", "scheduled_boundary", "id"
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    paper_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(PaperTradingRunRecord.id),
        nullable=False,
    )
    asset: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration_version: Mapped[int] = mapped_column(Integer, nullable=False)
    scheduled_boundary: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_score_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey(MarketScoreRecord.id)
    )
    decision_snapshot_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey(DecisionSnapshotRecord.id)
    )
    paper_trade_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey(PaperTradeRecord.id)
    )
    result: Mapped[str | None] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False)


@dataclass(frozen=True, slots=True)
class Instrument:
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval


MVP_INSTRUMENTS = (
    Instrument(
        Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
    ),
    Instrument(
        Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
    ),
)


def latest_completed_boundary(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("execution time must be timezone-aware")
    utc = moment.astimezone(UTC)
    minute = 30 if utc.minute >= 30 else 0
    return utc.replace(minute=minute, second=0, microsecond=0)


async def claim_cycle(
    *,
    session: AsyncSession,
    run_id: UUID,
    instrument: Instrument,
    boundary: datetime,
    started_at: datetime,
    strategy_name: str,
    strategy_version: int,
    configuration_version: int,
) -> PaperDecisionCycleRecord | None:
    cycle_id = uuid4()
    statement = (
        insert(PaperDecisionCycleRecord)
        .values(
            id=cycle_id,
            paper_run_id=run_id,
            asset=instrument.asset.value,
            provider=instrument.provider.value,
            market_type=instrument.market_type.value,
            interval=instrument.interval.value,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            configuration_version=configuration_version,
            scheduled_boundary=boundary,
            source_score_id=None,
            decision_snapshot_id=None,
            paper_trade_id=None,
            result=None,
            started_at=started_at,
            completed_at=None,
            status=DecisionCycleStatus.PROCESSING.value,
        )
        .on_conflict_do_nothing(constraint="uq_paper_cycles_natural_identity")
        .returning(PaperDecisionCycleRecord.id)
    )
    claimed = await session.scalar(statement)
    if claimed is None:
        return None
    return cast(
        PaperDecisionCycleRecord | None,
        await session.get(PaperDecisionCycleRecord, cycle_id),
    )


async def get_cycle_by_identity(
    *,
    session: AsyncSession,
    run_id: UUID,
    instrument: Instrument,
    boundary: datetime,
    strategy_name: str,
    strategy_version: int,
    configuration_version: int,
) -> PaperDecisionCycleRecord | None:
    return cast(
        PaperDecisionCycleRecord | None,
        await session.scalar(
            select(PaperDecisionCycleRecord).where(
                PaperDecisionCycleRecord.paper_run_id == run_id,
                PaperDecisionCycleRecord.asset == instrument.asset.value,
                PaperDecisionCycleRecord.provider == instrument.provider.value,
                PaperDecisionCycleRecord.market_type == instrument.market_type.value,
                PaperDecisionCycleRecord.interval == instrument.interval.value,
                PaperDecisionCycleRecord.strategy_name == strategy_name,
                PaperDecisionCycleRecord.strategy_version == strategy_version,
                PaperDecisionCycleRecord.configuration_version == configuration_version,
                PaperDecisionCycleRecord.scheduled_boundary == boundary,
            )
        ),
    )


async def get_boundary_score_evidence(
    *,
    session: AsyncSession,
    instrument: Instrument,
    boundary: datetime,
    run_started_at: datetime,
) -> ScoreEvidence:
    row = (
        await session.execute(
            select(
                MarketScoreRecord,
                MarketCandleRecord.close_price,
                MarketCandleRecord.close_time,
            )
            .join(
                MarketCandleRecord,
                (MarketCandleRecord.asset == MarketScoreRecord.asset)
                & (MarketCandleRecord.provider == MarketScoreRecord.provider)
                & (MarketCandleRecord.market_type == MarketScoreRecord.market_type)
                & (MarketCandleRecord.interval == MarketScoreRecord.interval)
                & (MarketCandleRecord.open_time == MarketScoreRecord.candle_open_time),
            )
            .where(
                MarketScoreRecord.asset == instrument.asset.value,
                MarketScoreRecord.provider == instrument.provider.value,
                MarketScoreRecord.market_type == instrument.market_type.value,
                MarketScoreRecord.interval == instrument.interval.value,
                MarketScoreRecord.model_name == REQUIRED_SCORE_MODEL_NAME,
                MarketScoreRecord.model_version == REQUIRED_SCORE_MODEL_VERSION,
                MarketScoreRecord.candle_open_time >= run_started_at,
                MarketCandleRecord.close_time <= boundary,
            )
            .order_by(MarketCandleRecord.close_time.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        raise DecisionEvidenceUnavailableError("eligible score evidence not found")
    score_record, close_price, close_time = row
    if boundary - close_time > MAX_EVIDENCE_AGE:
        raise DecisionEvidenceUnavailableError("eligible score evidence is stale")
    from app.scores.models import MarketScore

    score = MarketScore.model_validate(
        {
            column.name: getattr(score_record, column.name)
            for column in score_record.__table__.columns
        }
    )
    return ScoreEvidence(score=score, source_market_price=close_price)


async def get_cycle(
    session: AsyncSession, cycle_id: UUID
) -> PaperDecisionCycleRecord | None:
    return await session.get(PaperDecisionCycleRecord, cycle_id)


async def list_cycles(
    *,
    session: AsyncSession,
    run_id: UUID | None,
    asset: Asset | None,
    limit: int,
    offset: int,
) -> list[PaperDecisionCycleRecord]:
    statement = (
        select(PaperDecisionCycleRecord)
        .order_by(
            PaperDecisionCycleRecord.scheduled_boundary.desc(),
            PaperDecisionCycleRecord.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )
    if run_id is not None:
        statement = statement.where(PaperDecisionCycleRecord.paper_run_id == run_id)
    if asset is not None:
        statement = statement.where(PaperDecisionCycleRecord.asset == asset.value)
    return list((await session.scalars(statement)).all())
