"""Create the market features table.

Revision ID: 20260716_02
Revises: 20260715_01
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260716_02"
down_revision = "20260715_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_features",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("candle_open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_name", sa.String(32), nullable=False),
        sa.Column("feature_version", sa.Integer(), nullable=False),
        sa.Column("value", sa.Numeric(38, 18), nullable=False),
        sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "asset IN ('BTC/USDT', 'ETH/USDT')", name="ck_market_features_asset"
        ),
        sa.CheckConstraint(
            "provider IN ('binance')", name="ck_market_features_provider"
        ),
        sa.CheckConstraint(
            "market_type IN ('spot')", name="ck_market_features_market_type"
        ),
        sa.CheckConstraint("interval IN ('5m')", name="ck_market_features_interval"),
        sa.CheckConstraint("feature_version > 0", name="ck_market_features_version"),
        sa.CheckConstraint(
            "feature_name IN ('ema_20')", name="ck_market_features_name"
        ),
        sa.UniqueConstraint(
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


def downgrade() -> None:
    op.drop_table("market_features")
