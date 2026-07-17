"""Create persisted market scores.

Revision ID: 20260716_04
Revises: 20260716_03
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260716_04"
down_revision = "20260716_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("candle_open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("calculated_at", sa.DateTime(timezone=True), nullable=False),
        *[
            sa.Column(name, sa.Numeric(38, 18), nullable=False)
            for name in (
                "trend_score",
                "momentum_score",
                "volume_activity",
                "volatility_ratio",
                "directional_score",
                "volume_confirmation_factor",
                "overall_score",
                "signal_strength",
                "trend_contribution",
                "rsi_contribution",
                "return_1_contribution",
                "return_12_contribution",
            )
        ],
        sa.Column("interpretation", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "asset IN ('BTC/USDT', 'ETH/USDT')", name="ck_market_scores_asset"
        ),
        sa.CheckConstraint("provider IN ('binance')", name="ck_market_scores_provider"),
        sa.CheckConstraint(
            "market_type IN ('spot')", name="ck_market_scores_market_type"
        ),
        sa.CheckConstraint("interval IN ('5m')", name="ck_market_scores_interval"),
        sa.CheckConstraint(
            "model_name IN ('baseline_technical')", name="ck_market_scores_model"
        ),
        sa.CheckConstraint("model_version > 0", name="ck_market_scores_version"),
        sa.CheckConstraint(
            "trend_score BETWEEN -1 AND 1", name="ck_market_scores_trend"
        ),
        sa.CheckConstraint(
            "momentum_score BETWEEN -1 AND 1", name="ck_market_scores_momentum"
        ),
        sa.CheckConstraint(
            "volume_activity BETWEEN 0 AND 1", name="ck_market_scores_volume"
        ),
        sa.CheckConstraint("volatility_ratio >= 0", name="ck_market_scores_volatility"),
        sa.CheckConstraint(
            "directional_score BETWEEN -1 AND 1", name="ck_market_scores_directional"
        ),
        sa.CheckConstraint(
            "volume_confirmation_factor BETWEEN 0.75 AND 1",
            name="ck_market_scores_confirmation",
        ),
        sa.CheckConstraint(
            "overall_score BETWEEN -1 AND 1", name="ck_market_scores_overall"
        ),
        sa.CheckConstraint(
            "signal_strength BETWEEN 0 AND 1", name="ck_market_scores_strength"
        ),
        sa.CheckConstraint(
            "signal_strength = abs(overall_score)",
            name="ck_market_scores_strength_matches",
        ),
        sa.CheckConstraint(
            "trend_contribution + rsi_contribution + return_1_contribution + "
            "return_12_contribution = directional_score",
            name="ck_market_scores_contributions",
        ),
        sa.CheckConstraint(
            "interpretation IN ('strongly_bearish', 'bearish', 'neutral', "
            "'bullish', 'strongly_bullish')",
            name="ck_market_scores_interpretation",
        ),
        sa.UniqueConstraint(
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


def downgrade() -> None:
    op.drop_table("market_scores")
