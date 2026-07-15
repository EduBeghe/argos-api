"""Create the market candles table.

Revision ID: 20260715_01
Revises:
Create Date: 2026-07-15
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260715_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create storage for normalized market candles."""

    decimal_type = sa.Numeric(precision=38, scale=18)
    op.create_table(
        "market_candles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("market_type", sa.String(length=32), nullable=False),
        sa.Column("interval", sa.String(length=16), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_price", decimal_type, nullable=False),
        sa.Column("high_price", decimal_type, nullable=False),
        sa.Column("low_price", decimal_type, nullable=False),
        sa.Column("close_price", decimal_type, nullable=False),
        sa.Column("base_asset_volume", decimal_type, nullable=False),
        sa.Column("quote_asset_volume", decimal_type, nullable=False),
        sa.Column("trade_count", sa.Integer(), nullable=False),
        sa.Column("taker_buy_base_asset_volume", decimal_type, nullable=False),
        sa.Column("taker_buy_quote_asset_volume", decimal_type, nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "asset IN ('BTC/USDT', 'ETH/USDT')", name="ck_market_candles_asset"
        ),
        sa.CheckConstraint(
            "provider IN ('binance')", name="ck_market_candles_provider"
        ),
        sa.CheckConstraint(
            "market_type IN ('spot')", name="ck_market_candles_market_type"
        ),
        sa.CheckConstraint("interval IN ('5m')", name="ck_market_candles_interval"),
        sa.CheckConstraint(
            "open_price > 0", name="ck_market_candles_open_price_positive"
        ),
        sa.CheckConstraint(
            "high_price > 0", name="ck_market_candles_high_price_positive"
        ),
        sa.CheckConstraint(
            "low_price > 0", name="ck_market_candles_low_price_positive"
        ),
        sa.CheckConstraint(
            "close_price > 0", name="ck_market_candles_close_price_positive"
        ),
        sa.CheckConstraint(
            "base_asset_volume >= 0",
            name="ck_market_candles_base_volume_nonnegative",
        ),
        sa.CheckConstraint(
            "quote_asset_volume >= 0",
            name="ck_market_candles_quote_volume_nonnegative",
        ),
        sa.CheckConstraint(
            "trade_count >= 0", name="ck_market_candles_trade_count_nonnegative"
        ),
        sa.CheckConstraint(
            "taker_buy_base_asset_volume >= 0",
            name="ck_market_candles_taker_base_nonnegative",
        ),
        sa.CheckConstraint(
            "taker_buy_quote_asset_volume >= 0",
            name="ck_market_candles_taker_quote_nonnegative",
        ),
        sa.CheckConstraint(
            "close_time >= open_time", name="ck_market_candles_time_order"
        ),
        sa.CheckConstraint(
            "high_price >= open_price", name="ck_market_candles_high_ge_open"
        ),
        sa.CheckConstraint(
            "high_price >= close_price", name="ck_market_candles_high_ge_close"
        ),
        sa.CheckConstraint(
            "high_price >= low_price", name="ck_market_candles_high_ge_low"
        ),
        sa.CheckConstraint(
            "low_price <= open_price", name="ck_market_candles_low_le_open"
        ),
        sa.CheckConstraint(
            "low_price <= close_price", name="ck_market_candles_low_le_close"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "asset",
            "provider",
            "market_type",
            "interval",
            "open_time",
            name="uq_market_candles_natural_identity",
        ),
    )


def downgrade() -> None:
    """Remove normalized market-candle storage."""

    op.drop_table("market_candles")
