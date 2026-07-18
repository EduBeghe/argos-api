"""Create persisted Paper Decision Cycles.

Revision ID: 20260718_07
Revises: 20260717_06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260718_07"
down_revision: str | None = "20260717_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "paper_decision_cycles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "paper_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_trading_runs.id"),
            nullable=False,
        ),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("strategy_version", sa.Integer(), nullable=False),
        sa.Column("configuration_version", sa.Integer(), nullable=False),
        sa.Column("scheduled_boundary", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "source_score_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("market_scores.id"),
        ),
        sa.Column(
            "decision_snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("decision_snapshots.id"),
        ),
        sa.Column(
            "paper_trade_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_trades.id"),
        ),
        sa.Column("result", sa.String(32)),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False),
        sa.CheckConstraint(
            "status IN ('processing', 'completed')", name="ck_paper_cycles_status"
        ),
        sa.CheckConstraint(
            "strategy_version > 0 AND configuration_version > 0",
            name="ck_paper_cycles_versions",
        ),
        sa.CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0 AND "
            "length(btrim(strategy_name)) > 0",
            name="ck_paper_cycles_identity",
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name="ck_paper_cycles_timestamps",
        ),
        sa.CheckConstraint(
            "(status = 'processing' AND completed_at IS NULL AND result IS NULL) OR "
            "(status = 'completed' AND completed_at IS NOT NULL AND "
            "result IS NOT NULL AND source_score_id IS NOT NULL AND "
            "decision_snapshot_id IS NOT NULL)",
            name="ck_paper_cycles_lifecycle",
        ),
        sa.UniqueConstraint(
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
    )
    op.create_index(
        "ix_paper_cycles_run_boundary",
        "paper_decision_cycles",
        ["paper_run_id", "scheduled_boundary", "id"],
    )


def downgrade() -> None:
    op.drop_table("paper_decision_cycles")
