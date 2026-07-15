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
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.market.models import MarketCandle

DECIMAL_PRECISION = 38
DECIMAL_SCALE = 18


class MarketCandleRecord(Base):
    """Database representation of a normalized market candle."""

    __tablename__ = "market_candles"
    __table_args__ = (
        UniqueConstraint(
            "asset",
            "provider",
            "market_type",
            "interval",
            "open_time",
            name="uq_market_candles_natural_identity",
        ),
        CheckConstraint(
            "asset IN ('BTC/USDT', 'ETH/USDT')", name="ck_market_candles_asset"
        ),
        CheckConstraint("provider IN ('binance')", name="ck_market_candles_provider"),
        CheckConstraint(
            "market_type IN ('spot')", name="ck_market_candles_market_type"
        ),
        CheckConstraint("interval IN ('5m')", name="ck_market_candles_interval"),
        CheckConstraint("open_price > 0", name="ck_market_candles_open_price_positive"),
        CheckConstraint("high_price > 0", name="ck_market_candles_high_price_positive"),
        CheckConstraint("low_price > 0", name="ck_market_candles_low_price_positive"),
        CheckConstraint(
            "close_price > 0", name="ck_market_candles_close_price_positive"
        ),
        CheckConstraint(
            "base_asset_volume >= 0", name="ck_market_candles_base_volume_nonnegative"
        ),
        CheckConstraint(
            "quote_asset_volume >= 0", name="ck_market_candles_quote_volume_nonnegative"
        ),
        CheckConstraint(
            "trade_count >= 0", name="ck_market_candles_trade_count_nonnegative"
        ),
        CheckConstraint(
            "taker_buy_base_asset_volume >= 0",
            name="ck_market_candles_taker_base_nonnegative",
        ),
        CheckConstraint(
            "taker_buy_quote_asset_volume >= 0",
            name="ck_market_candles_taker_quote_nonnegative",
        ),
        CheckConstraint("close_time >= open_time", name="ck_market_candles_time_order"),
        CheckConstraint(
            "high_price >= open_price", name="ck_market_candles_high_ge_open"
        ),
        CheckConstraint(
            "high_price >= close_price", name="ck_market_candles_high_ge_close"
        ),
        CheckConstraint(
            "high_price >= low_price", name="ck_market_candles_high_ge_low"
        ),
        CheckConstraint(
            "low_price <= open_price", name="ck_market_candles_low_le_open"
        ),
        CheckConstraint(
            "low_price <= close_price", name="ck_market_candles_low_le_close"
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    open_price: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    high_price: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    low_price: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    close_price: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    base_asset_volume: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    quote_asset_volume: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    taker_buy_base_asset_volume: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    taker_buy_quote_asset_volume: Mapped[Decimal] = mapped_column(
        Numeric(DECIMAL_PRECISION, DECIMAL_SCALE), nullable=False
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


def to_record(candle: MarketCandle) -> MarketCandleRecord:
    """Map a domain candle to its persistence representation."""

    return MarketCandleRecord(**candle.model_dump(mode="python"))


def to_domain(record: MarketCandleRecord) -> MarketCandle:
    """Map a persisted candle to its domain representation."""

    return MarketCandle.model_validate(
        {
            column.name: getattr(record, column.name)
            for column in record.__table__.columns
        }
    )
