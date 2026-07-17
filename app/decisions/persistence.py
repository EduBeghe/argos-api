from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    select,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.decisions.models import DecisionSnapshot
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord
from app.scores.models import MarketScore
from app.scores.persistence import MarketScoreRecord
from app.strategies.baseline_long_only import (
    REQUIRED_SCORE_MODEL_NAME,
    REQUIRED_SCORE_MODEL_VERSION,
)


class DecisionSnapshotRecord(Base):
    __tablename__ = "decision_snapshots"
    __table_args__ = (
        CheckConstraint("length(btrim(asset)) > 0", name="ck_decisions_asset"),
        CheckConstraint("length(btrim(provider)) > 0", name="ck_decisions_provider"),
        CheckConstraint("length(btrim(market_type)) > 0", name="ck_decisions_market"),
        CheckConstraint("length(btrim(interval)) > 0", name="ck_decisions_interval"),
        CheckConstraint(
            "length(btrim(source_score_model_name)) > 0",
            name="ck_decisions_score_model",
        ),
        CheckConstraint(
            "source_score_model_version > 0", name="ck_decisions_score_version"
        ),
        CheckConstraint(
            "length(btrim(strategy_name)) > 0", name="ck_decisions_strategy"
        ),
        CheckConstraint("strategy_version > 0", name="ck_decisions_strategy_version"),
        CheckConstraint(
            "configuration_version > 0", name="ck_decisions_configuration_version"
        ),
        CheckConstraint(
            "source_market_price > 0", name="ck_decisions_source_market_price"
        ),
        CheckConstraint(
            "source_overall_score BETWEEN -1 AND 1",
            name="ck_decisions_source_overall_score",
        ),
        CheckConstraint(
            "action IN ('open_position', 'increase_position', 'sell_partial', "
            "'sell_all', 'no_action')",
            name="ck_decisions_action",
        ),
        CheckConstraint(
            "requested_target_portfolio_percentage IS NULL OR "
            "requested_target_portfolio_percentage > 0 AND "
            "requested_target_portfolio_percentage <= 100",
            name="ck_decisions_target_percentage",
        ),
        CheckConstraint(
            "requested_position_percentage IS NULL OR "
            "requested_position_percentage > 0 AND "
            "requested_position_percentage <= 100",
            name="ck_decisions_position_percentage",
        ),
        CheckConstraint(
            "(action = 'open_position' AND "
            "requested_target_portfolio_percentage IS NOT NULL AND "
            "requested_position_percentage IS NULL) OR "
            "(action = 'increase_position' AND "
            "requested_target_portfolio_percentage IS NOT NULL AND "
            "requested_position_percentage IS NULL) OR "
            "(action = 'sell_partial' AND "
            "requested_target_portfolio_percentage IS NULL AND "
            "requested_position_percentage < 100) OR "
            "(action = 'sell_all' AND "
            "requested_target_portfolio_percentage IS NULL AND "
            "requested_position_percentage = 100) OR "
            "(action = 'no_action' AND "
            "requested_target_portfolio_percentage IS NULL AND "
            "requested_position_percentage IS NULL)",
            name="ck_decisions_action_sizing",
        ),
        CheckConstraint(
            "length(btrim(applied_rule)) > 0", name="ck_decisions_applied_rule"
        ),
        CheckConstraint(
            "length(btrim(explanation)) > 0", name="ck_decisions_explanation"
        ),
        UniqueConstraint(
            "source_score_id",
            "strategy_name",
            "strategy_version",
            "configuration_version",
            "position_is_open",
            name="uq_decision_snapshots_source_evaluation",
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    asset: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    decision_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_candle_open_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_market_price: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    source_score_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("market_scores.id"),
        nullable=False,
    )
    source_score_model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    source_score_model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    source_overall_score: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration_version: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_target_portfolio_percentage: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18), nullable=True
    )
    requested_position_percentage: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18), nullable=True
    )
    applied_rule: Mapped[str] = mapped_column(String(64), nullable=False)
    explanation: Mapped[str] = mapped_column(String(512), nullable=False)
    position_is_open: Mapped[bool] = mapped_column(Boolean, nullable=False)


@dataclass(frozen=True, slots=True)
class ScoreEvidence:
    score: MarketScore
    source_market_price: Decimal


async def insert_decision_snapshot(
    session: AsyncSession, snapshot: DecisionSnapshot
) -> bool:
    statement = (
        insert(DecisionSnapshotRecord)
        .values(snapshot.model_dump(mode="python"))
        .on_conflict_do_nothing(
            index_elements=[
                "source_score_id",
                "strategy_name",
                "strategy_version",
                "configuration_version",
                "position_is_open",
            ]
        )
        .returning(DecisionSnapshotRecord.id)
    )
    return await session.scalar(statement) is not None


async def get_latest_score_evidence(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
) -> ScoreEvidence | None:
    row = (
        await session.execute(
            select(MarketScoreRecord, MarketCandleRecord.close_price)
            .join(
                MarketCandleRecord,
                (MarketCandleRecord.asset == MarketScoreRecord.asset)
                & (MarketCandleRecord.provider == MarketScoreRecord.provider)
                & (MarketCandleRecord.market_type == MarketScoreRecord.market_type)
                & (MarketCandleRecord.interval == MarketScoreRecord.interval)
                & (MarketCandleRecord.open_time == MarketScoreRecord.candle_open_time),
            )
            .where(
                MarketScoreRecord.asset == asset.value,
                MarketScoreRecord.provider == provider.value,
                MarketScoreRecord.market_type == market_type.value,
                MarketScoreRecord.interval == interval.value,
                MarketScoreRecord.model_name == REQUIRED_SCORE_MODEL_NAME,
                MarketScoreRecord.model_version == REQUIRED_SCORE_MODEL_VERSION,
            )
            .order_by(MarketScoreRecord.candle_open_time.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        return None
    record, close_price = row
    score = MarketScore.model_validate(
        {
            column.name: getattr(record, column.name)
            for column in record.__table__.columns
        }
    )
    return ScoreEvidence(score=score, source_market_price=close_price)


async def get_latest_decision_snapshot(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
) -> DecisionSnapshot | None:
    record = await session.scalar(
        select(DecisionSnapshotRecord)
        .where(
            DecisionSnapshotRecord.asset == asset.value,
            DecisionSnapshotRecord.provider == provider.value,
            DecisionSnapshotRecord.market_type == market_type.value,
            DecisionSnapshotRecord.interval == interval.value,
        )
        .order_by(
            DecisionSnapshotRecord.decision_timestamp.desc(),
            DecisionSnapshotRecord.source_candle_open_time.desc(),
        )
        .limit(1)
    )
    if record is None:
        return None
    return DecisionSnapshot.model_validate(
        {
            column.name: getattr(record, column.name)
            for column in record.__table__.columns
        }
    )
