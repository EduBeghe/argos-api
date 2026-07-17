"""Create persisted Paper Trading runs, positions, and trades.

Revision ID: 20260717_06
Revises: 20260717_05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260717_06"
down_revision: str | None = "20260717_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    numeric = sa.Numeric(38, 18)
    op.create_table(
        "paper_trading_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("account_currency", sa.String(16), nullable=False),
        sa.Column("initial_cash", numeric, nullable=False),
        sa.Column("current_cash", numeric, nullable=False),
        sa.Column("configuration_version", sa.Integer(), nullable=False),
        sa.Column("fee_rate", numeric, nullable=False),
        sa.Column("slippage_rate", numeric, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_at", sa.DateTime(timezone=True)),
        sa.Column("stop_reason", sa.String(512)),
        sa.Column(
            "predecessor_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_trading_runs.id"),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'stopped')", name="ck_paper_runs_status"
        ),
        sa.CheckConstraint(
            "length(btrim(account_currency)) > 0", name="ck_paper_runs_currency"
        ),
        sa.CheckConstraint("initial_cash > 0", name="ck_paper_runs_initial_cash"),
        sa.CheckConstraint("current_cash >= 0", name="ck_paper_runs_current_cash"),
        sa.CheckConstraint(
            "fee_rate >= 0 AND fee_rate < 1", name="ck_paper_runs_fee_rate"
        ),
        sa.CheckConstraint(
            "slippage_rate >= 0 AND slippage_rate < 1",
            name="ck_paper_runs_slippage_rate",
        ),
        sa.CheckConstraint(
            "(status = 'active' AND stopped_at IS NULL) OR "
            "(status = 'stopped' AND stopped_at IS NOT NULL)",
            name="ck_paper_runs_lifecycle",
        ),
        sa.CheckConstraint("configuration_version > 0", name="ck_paper_runs_version"),
        sa.CheckConstraint(
            "predecessor_run_id IS NULL OR predecessor_run_id <> id",
            name="ck_paper_runs_predecessor",
        ),
        sa.CheckConstraint(
            "stopped_at IS NULL OR stopped_at >= started_at",
            name="ck_paper_runs_timestamps",
        ),
    )
    op.create_index(
        "uq_paper_runs_active_currency",
        "paper_trading_runs",
        ["account_currency"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "paper_positions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_trading_runs.id"),
            nullable=False,
        ),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("quantity", numeric, nullable=False),
        sa.Column("average_entry_price", numeric, nullable=False),
        sa.Column("total_entry_notional", numeric, nullable=False),
        sa.Column("accumulated_entry_fees", numeric, nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.Column("exit_price", numeric),
        sa.Column("gross_realized_pnl", numeric),
        sa.Column("net_realized_pnl", numeric),
        sa.CheckConstraint(
            "status IN ('open', 'closed')", name="ck_paper_positions_status"
        ),
        sa.CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0",
            name="ck_paper_positions_identity",
        ),
        sa.CheckConstraint(
            "quantity >= 0 AND average_entry_price > 0 AND "
            "total_entry_notional > 0 AND accumulated_entry_fees >= 0",
            name="ck_paper_positions_amounts",
        ),
        sa.CheckConstraint(
            "(status = 'open' AND quantity > 0 AND closed_at IS NULL AND "
            "exit_price IS NULL AND gross_realized_pnl IS NULL AND "
            "net_realized_pnl IS NULL) OR (status = 'closed' AND quantity = 0 AND "
            "closed_at IS NOT NULL AND exit_price > 0 AND "
            "gross_realized_pnl IS NOT NULL AND net_realized_pnl IS NOT NULL)",
            name="ck_paper_positions_lifecycle",
        ),
        sa.CheckConstraint(
            "closed_at IS NULL OR closed_at >= opened_at",
            name="ck_paper_positions_timestamps",
        ),
    )
    op.create_index(
        "uq_paper_positions_open_identity",
        "paper_positions",
        ["run_id", "asset", "provider", "market_type", "interval"],
        unique=True,
        postgresql_where=sa.text("status = 'open'"),
    )
    op.create_table(
        "paper_trades",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_trading_runs.id"),
            nullable=False,
        ),
        sa.Column(
            "decision_snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("decision_snapshots.id"),
            nullable=False,
        ),
        sa.Column(
            "position_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_positions.id"),
            nullable=False,
        ),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("observed_market_price", numeric, nullable=False),
        sa.Column("slippage_rate", numeric, nullable=False),
        sa.Column("execution_price", numeric, nullable=False),
        sa.Column("quantity", numeric, nullable=False),
        sa.Column("executed_notional", numeric, nullable=False),
        sa.Column("fee_rate", numeric, nullable=False),
        sa.Column("fee_amount", numeric, nullable=False),
        sa.Column("cash_impact", numeric, nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("configuration_version", sa.Integer(), nullable=False),
        sa.Column("equity_before", numeric),
        sa.CheckConstraint("side IN ('buy', 'sell')", name="ck_paper_trades_side"),
        sa.CheckConstraint(
            "length(btrim(asset)) > 0 AND length(btrim(provider)) > 0 AND "
            "length(btrim(market_type)) > 0 AND length(btrim(interval)) > 0",
            name="ck_paper_trades_identity",
        ),
        sa.CheckConstraint(
            "observed_market_price > 0 AND execution_price > 0 AND quantity > 0 AND "
            "executed_notional > 0 AND fee_amount >= 0",
            name="ck_paper_trades_amounts",
        ),
        sa.CheckConstraint(
            "fee_rate >= 0 AND fee_rate < 1 AND slippage_rate >= 0 AND "
            "slippage_rate < 1",
            name="ck_paper_trades_rates",
        ),
        sa.CheckConstraint("configuration_version > 0", name="ck_paper_trades_version"),
        sa.CheckConstraint(
            "(side = 'buy' AND cash_impact < 0) OR (side = 'sell' AND cash_impact > 0)",
            name="ck_paper_trades_cash_direction",
        ),
    )
    op.create_index(
        "uq_paper_trades_decision",
        "paper_trades",
        ["decision_snapshot_id"],
        unique=True,
    )
    op.create_index(
        "ix_paper_trades_run_executed", "paper_trades", ["run_id", "executed_at", "id"]
    )


def downgrade() -> None:
    op.drop_table("paper_trades")
    op.drop_table("paper_positions")
    op.drop_table("paper_trading_runs")
