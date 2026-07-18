"""Create immutable Paper portfolio and Position snapshots.

Revision ID: 20260718_08
Revises: 20260718_07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260718_08"
down_revision: str | None = "20260718_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    numeric = sa.Numeric(38, 18)
    op.create_table(
        "paper_portfolio_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_trading_runs.id"),
            nullable=False,
        ),
        sa.Column("scheduled_boundary", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("account_currency", sa.String(16), nullable=False),
        sa.Column("initial_cash", numeric, nullable=False),
        sa.Column("current_cash", numeric, nullable=False),
        sa.Column("positions_market_value", numeric, nullable=False),
        sa.Column("equity", numeric, nullable=False),
        sa.Column("total_pnl", numeric, nullable=False),
        sa.Column("total_return", numeric),
        sa.Column("configuration_version", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "length(btrim(account_currency)) > 0",
            name="ck_portfolio_snapshots_currency",
        ),
        sa.CheckConstraint(
            "configuration_version > 0", name="ck_portfolio_snapshots_version"
        ),
        sa.CheckConstraint(
            "recorded_at >= scheduled_boundary",
            name="ck_portfolio_snapshots_timestamps",
        ),
        sa.UniqueConstraint(
            "run_id", "scheduled_boundary", name="uq_portfolio_snapshots_run_boundary"
        ),
    )
    op.create_index(
        "ix_portfolio_snapshots_run_boundary",
        "paper_portfolio_snapshots",
        ["run_id", "scheduled_boundary", "id"],
    )
    op.create_table(
        "paper_position_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_portfolio_snapshots.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("quantity", numeric, nullable=False),
        sa.Column("average_entry_price", numeric, nullable=False),
        sa.Column("cost_basis", numeric, nullable=False),
        sa.Column("mark_price", numeric, nullable=False),
        sa.Column("market_value", numeric, nullable=False),
        sa.Column("unrealized_pnl", numeric, nullable=False),
        sa.Column("unrealized_return", numeric),
        sa.Column("mark_candle_open_time", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0",
            name="ck_position_snapshots_identity",
        ),
        sa.CheckConstraint(
            "quantity > 0 AND average_entry_price > 0 AND cost_basis > 0 AND "
            "mark_price > 0 AND market_value > 0",
            name="ck_position_snapshots_amounts",
        ),
        sa.UniqueConstraint(
            "portfolio_snapshot_id",
            "position_id",
            name="uq_position_snapshots_parent_position",
        ),
    )
    op.create_index(
        "ix_position_snapshots_parent",
        "paper_position_snapshots",
        ["portfolio_snapshot_id", "id"],
    )


def downgrade() -> None:
    op.drop_table("paper_position_snapshots")
    op.drop_table("paper_portfolio_snapshots")
