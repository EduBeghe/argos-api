"""Expand supported analytical feature names.

Revision ID: 20260716_03
Revises: 20260716_02
"""

from alembic import op

revision = "20260716_03"
down_revision = "20260716_02"
branch_labels = None
depends_on = None

EXPANDED = (
    "feature_name IN ('ema_20', 'ema_50', 'rsi_14', 'atr_14', "
    "'return_1', 'return_12', 'relative_volume_20')"
)


def upgrade() -> None:
    op.drop_constraint("ck_market_features_name", "market_features", type_="check")
    op.create_check_constraint("ck_market_features_name", "market_features", EXPANDED)


def downgrade() -> None:
    """Delete new-feature rows before restoring the EMA-20-only constraint."""
    op.drop_constraint("ck_market_features_name", "market_features", type_="check")
    op.execute("DELETE FROM market_features WHERE feature_name <> 'ema_20'")
    op.create_check_constraint(
        "ck_market_features_name", "market_features", "feature_name IN ('ema_20')"
    )
