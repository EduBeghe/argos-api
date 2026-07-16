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
from app.features.models import FeatureName, MarketFeature
from app.market.enums import Asset, Interval, MarketType, Provider


class MarketFeatureRecord(Base):
    __tablename__ = "market_features"
    __table_args__ = (
        CheckConstraint(
            "asset IN ('BTC/USDT', 'ETH/USDT')", name="ck_market_features_asset"
        ),
        CheckConstraint("provider IN ('binance')", name="ck_market_features_provider"),
        CheckConstraint(
            "market_type IN ('spot')", name="ck_market_features_market_type"
        ),
        CheckConstraint("interval IN ('5m')", name="ck_market_features_interval"),
        CheckConstraint("feature_version > 0", name="ck_market_features_version"),
        CheckConstraint("feature_name IN ('ema_20')", name="ck_market_features_name"),
        UniqueConstraint(
            "asset",
            "provider",
            "market_type",
            "interval",
            "candle_open_time",
            "feature_name",
            "feature_version",
            name="uq_market_features_identity",
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
    feature_name: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_version: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


async def insert_market_features(
    session: AsyncSession, features: Sequence[MarketFeature]
) -> int:
    if not features:
        return 0
    statement = (
        insert(MarketFeatureRecord)
        .values([feature.model_dump(mode="python") for feature in features])
        .on_conflict_do_nothing(
            index_elements=[
                "asset",
                "provider",
                "market_type",
                "interval",
                "candle_open_time",
                "feature_name",
                "feature_version",
            ]
        )
        .returning(MarketFeatureRecord.id)
    )
    return len((await session.scalars(statement)).all())


async def get_latest_feature(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    feature_name: FeatureName,
    feature_version: int,
) -> MarketFeature | None:
    statement = (
        select(MarketFeatureRecord)
        .where(
            MarketFeatureRecord.asset == asset.value,
            MarketFeatureRecord.provider == provider.value,
            MarketFeatureRecord.market_type == market_type.value,
            MarketFeatureRecord.interval == interval.value,
            MarketFeatureRecord.feature_name == feature_name.value,
            MarketFeatureRecord.feature_version == feature_version,
        )
        .order_by(MarketFeatureRecord.candle_open_time.desc())
        .limit(1)
    )
    record = await session.scalar(statement)
    if record is None:
        return None
    return MarketFeature.model_validate(
        {
            column.name: getattr(record, column.name)
            for column in record.__table__.columns
        }
    )
