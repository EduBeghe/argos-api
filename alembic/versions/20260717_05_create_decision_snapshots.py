"""Create immutable Decision Snapshots.

Revision ID: 20260717_05
Revises: 20260716_04
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260717_05"
down_revision = "20260716_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "decision_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("decision_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "source_candle_open_time", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("source_market_price", sa.Numeric(38, 18), nullable=False),
        sa.Column("source_score_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_score_model_name", sa.String(64), nullable=False),
        sa.Column("source_score_model_version", sa.Integer(), nullable=False),
        sa.Column("source_overall_score", sa.Numeric(38, 18), nullable=False),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("strategy_version", sa.Integer(), nullable=False),
        sa.Column("configuration_version", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column(
            "requested_target_portfolio_percentage",
            sa.Numeric(38, 18),
            nullable=True,
        ),
        sa.Column("requested_position_percentage", sa.Numeric(38, 18), nullable=True),
        sa.Column("applied_rule", sa.String(64), nullable=False),
        sa.Column("explanation", sa.String(512), nullable=False),
        sa.Column("position_is_open", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["source_score_id"], ["market_scores.id"]),
        sa.CheckConstraint("length(btrim(asset)) > 0", name="ck_decisions_asset"),
        sa.CheckConstraint("length(btrim(provider)) > 0", name="ck_decisions_provider"),
        sa.CheckConstraint(
            "length(btrim(market_type)) > 0", name="ck_decisions_market"
        ),
        sa.CheckConstraint("length(btrim(interval)) > 0", name="ck_decisions_interval"),
        sa.CheckConstraint(
            "length(btrim(source_score_model_name)) > 0",
            name="ck_decisions_score_model",
        ),
        sa.CheckConstraint(
            "source_score_model_version > 0", name="ck_decisions_score_version"
        ),
        sa.CheckConstraint(
            "length(btrim(strategy_name)) > 0", name="ck_decisions_strategy"
        ),
        sa.CheckConstraint(
            "strategy_version > 0", name="ck_decisions_strategy_version"
        ),
        sa.CheckConstraint(
            "configuration_version > 0", name="ck_decisions_configuration_version"
        ),
        sa.CheckConstraint(
            "source_market_price > 0", name="ck_decisions_source_market_price"
        ),
        sa.CheckConstraint(
            "source_overall_score BETWEEN -1 AND 1",
            name="ck_decisions_source_overall_score",
        ),
        sa.CheckConstraint(
            "action IN ('open_position', 'increase_position', 'sell_partial', "
            "'sell_all', 'no_action')",
            name="ck_decisions_action",
        ),
        sa.CheckConstraint(
            "requested_target_portfolio_percentage IS NULL OR "
            "requested_target_portfolio_percentage > 0 AND "
            "requested_target_portfolio_percentage <= 100",
            name="ck_decisions_target_percentage",
        ),
        sa.CheckConstraint(
            "requested_position_percentage IS NULL OR "
            "requested_position_percentage > 0 AND "
            "requested_position_percentage <= 100",
            name="ck_decisions_position_percentage",
        ),
        sa.CheckConstraint(
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
        sa.CheckConstraint(
            "(action = 'open_position' AND NOT position_is_open AND "
            "applied_rule = 'entry_threshold_met') OR "
            "(action IN ('increase_position', 'sell_partial', 'sell_all') AND "
            "position_is_open) OR "
            "(action = 'no_action' AND NOT position_is_open AND "
            "applied_rule = 'entry_threshold_not_met') OR "
            "(action = 'no_action' AND position_is_open AND "
            "applied_rule = 'open_position_retained')",
            name="ck_decisions_action_state_rule",
        ),
        sa.CheckConstraint(
            "length(btrim(applied_rule)) > 0", name="ck_decisions_applied_rule"
        ),
        sa.CheckConstraint(
            "length(btrim(explanation)) > 0", name="ck_decisions_explanation"
        ),
        sa.UniqueConstraint(
            "source_score_id",
            "strategy_name",
            "strategy_version",
            "configuration_version",
            "position_is_open",
            name="uq_decision_snapshots_source_evaluation",
        ),
    )


def downgrade() -> None:
    op.drop_table("decision_snapshots")
