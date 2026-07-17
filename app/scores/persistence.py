from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
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
from app.market.enums import Asset, Interval, MarketType, Provider
from app.scores.models import MarketScore


class MarketScoreRecord(Base):
    __tablename__ = "market_scores"
    __table_args__ = (
        CheckConstraint(
            "asset IN ('BTC/USDT', 'ETH/USDT')", name="ck_market_scores_asset"
        ),
        CheckConstraint("provider IN ('binance')", name="ck_market_scores_provider"),
        CheckConstraint("market_type IN ('spot')", name="ck_market_scores_market_type"),
        CheckConstraint("interval IN ('5m')", name="ck_market_scores_interval"),
        CheckConstraint(
            "model_name IN ('baseline_technical')", name="ck_market_scores_model"
        ),
        CheckConstraint("model_version > 0", name="ck_market_scores_version"),
        CheckConstraint("trend_score BETWEEN -1 AND 1", name="ck_market_scores_trend"),
        CheckConstraint(
            "momentum_score BETWEEN -1 AND 1", name="ck_market_scores_momentum"
        ),
        CheckConstraint(
            "volume_activity BETWEEN 0 AND 1", name="ck_market_scores_volume"
        ),
        CheckConstraint("volatility_ratio >= 0", name="ck_market_scores_volatility"),
        CheckConstraint(
            "directional_score BETWEEN -1 AND 1", name="ck_market_scores_directional"
        ),
        CheckConstraint(
            "volume_confirmation_factor BETWEEN 0.75 AND 1",
            name="ck_market_scores_confirmation",
        ),
        CheckConstraint(
            "overall_score BETWEEN -1 AND 1", name="ck_market_scores_overall"
        ),
        CheckConstraint(
            "signal_strength BETWEEN 0 AND 1", name="ck_market_scores_strength"
        ),
        CheckConstraint(
            "signal_strength = abs(overall_score)",
            name="ck_market_scores_strength_matches",
        ),
        CheckConstraint(
            "trend_contribution + rsi_contribution + return_1_contribution + "
            "return_12_contribution = directional_score",
            name="ck_market_scores_contributions",
        ),
        CheckConstraint(
            "interpretation IN ('strongly_bearish', 'bearish', 'neutral', "
            "'bullish', 'strongly_bullish')",
            name="ck_market_scores_interpretation",
        ),
        UniqueConstraint(
            "asset",
            "provider",
            "market_type",
            "interval",
            "candle_open_time",
            "model_name",
            "model_version",
            name="uq_market_scores_identity",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    candle_open_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    trend_score: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    momentum_score: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    volume_activity: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    volatility_ratio: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    directional_score: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    volume_confirmation_factor: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    overall_score: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    signal_strength: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    trend_contribution: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    rsi_contribution: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    return_1_contribution: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    return_12_contribution: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    interpretation: Mapped[str] = mapped_column(String(32), nullable=False)


async def insert_market_scores(
    session: AsyncSession, scores: Sequence[MarketScore]
) -> int:
    if not scores:
        return 0
    statement = (
        insert(MarketScoreRecord)
        .values([score.model_dump(mode="python") for score in scores])
        .on_conflict_do_nothing(
            index_elements=[
                "asset",
                "provider",
                "market_type",
                "interval",
                "candle_open_time",
                "model_name",
                "model_version",
            ]
        )
        .returning(MarketScoreRecord.id)
    )
    return len((await session.scalars(statement)).all())


async def get_latest_market_score(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    model_name: str,
    model_version: int,
) -> MarketScore | None:
    record = await session.scalar(
        select(MarketScoreRecord)
        .where(
            MarketScoreRecord.asset == asset.value,
            MarketScoreRecord.provider == provider.value,
            MarketScoreRecord.market_type == market_type.value,
            MarketScoreRecord.interval == interval.value,
            MarketScoreRecord.model_name == model_name,
            MarketScoreRecord.model_version == model_version,
        )
        .order_by(MarketScoreRecord.candle_open_time.desc())
        .limit(1)
    )
    if record is None:
        return None
    return MarketScore.model_validate(
        {
            column.name: getattr(record, column.name)
            for column in record.__table__.columns
        }
    )
