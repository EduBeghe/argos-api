"""Add durable ML refinement worker leases.

Revision ID: 20260729_24
Revises: 20260729_23
"""

# ruff: noqa: E501

import sqlalchemy as sa

from alembic import op

revision = "20260729_24"
down_revision = "20260729_23"
branch_labels = None
depends_on = None


def _new_lifecycle() -> str:
    return (
        "(status = 'queued' AND started_at IS NULL AND completed_at IS NULL "
        "AND failed_at IS NULL AND worker_id IS NULL AND claimed_at IS NULL "
        "AND heartbeat_at IS NULL) OR "
        "(status = 'running' AND started_at IS NOT NULL AND completed_at IS NULL "
        "AND failed_at IS NULL AND worker_id IS NOT NULL AND claimed_at IS NOT NULL "
        "AND heartbeat_at IS NOT NULL) OR "
        "(status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
        "AND failed_at IS NULL AND worker_id IS NULL AND claimed_at IS NULL "
        "AND heartbeat_at IS NULL) OR "
        "(status = 'failed' AND started_at IS NOT NULL AND completed_at IS NULL "
        "AND failed_at IS NOT NULL AND worker_id IS NULL AND claimed_at IS NULL "
        "AND heartbeat_at IS NULL) OR "
        "(status = 'cancelled' AND completed_at IS NOT NULL AND failed_at IS NULL "
        "AND worker_id IS NULL AND claimed_at IS NULL AND heartbeat_at IS NULL)"
    )


def upgrade() -> None:
    op.add_column(
        "ml_refinement_runs",
        sa.Column("evaluations_selected", sa.Integer(), nullable=True),
    )
    for name, column in (
        ("worker_id", sa.String(128)),
        ("claimed_at", sa.DateTime(timezone=True)),
        ("heartbeat_at", sa.DateTime(timezone=True)),
        ("cancellation_requested_at", sa.DateTime(timezone=True)),
    ):
        op.add_column("ml_refinement_runs", sa.Column(name, column))
    op.execute(
        "UPDATE ml_refinement_runs SET evaluations_selected = "
        "LEAST(maximum_trials, combinations_planned)"
    )
    op.alter_column("ml_refinement_runs", "evaluations_selected", nullable=False)
    op.drop_constraint(
        "ck_ml_refinement_run_lifecycle", "ml_refinement_runs", type_="check"
    )
    op.drop_constraint(
        "ck_ml_refinement_run_budgets", "ml_refinement_runs", type_="check"
    )
    # Existing running rows predate leases. Mark them failed and explicitly retryable.
    op.execute(
        "UPDATE ml_refinement_runs SET status='failed', failed_at=NOW(), "
        "failure_type='WorkerLeaseExpired', "
        "failure_message='running row migrated without a durable worker lease' "
        "WHERE status='running'"
    )
    op.create_check_constraint(
        "ck_ml_refinement_run_lifecycle", "ml_refinement_runs", _new_lifecycle()
    )
    op.create_check_constraint(
        "ck_ml_refinement_run_budgets",
        "ml_refinement_runs",
        "maximum_trials > 0 AND maximum_runtime_seconds > 0 "
        "AND combinations_planned >= 0 AND evaluations_computed >= 0 "
        "AND evaluations_reused >= 0 AND evaluations_omitted >= 0 "
        "AND evaluations_selected >= 0 "
        "AND evaluations_computed + evaluations_reused <= evaluations_selected",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ml_refinement_run_lifecycle", "ml_refinement_runs", type_="check"
    )
    op.drop_constraint(
        "ck_ml_refinement_run_budgets", "ml_refinement_runs", type_="check"
    )
    op.execute(
        "UPDATE ml_refinement_runs SET status='failed', failed_at=COALESCE(failed_at, NOW()), "
        "completed_at=NULL WHERE status='cancelled' AND started_at IS NULL"
    )
    op.create_check_constraint(
        "ck_ml_refinement_run_lifecycle",
        "ml_refinement_runs",
        "(status = 'queued' AND started_at IS NULL AND completed_at IS NULL AND failed_at IS NULL) OR "
        "(status = 'running' AND started_at IS NOT NULL AND completed_at IS NULL AND failed_at IS NULL) OR "
        "(status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL AND failed_at IS NULL) OR "
        "(status = 'failed' AND started_at IS NOT NULL AND completed_at IS NULL AND failed_at IS NOT NULL) OR "
        "(status = 'cancelled' AND started_at IS NOT NULL AND completed_at IS NOT NULL AND failed_at IS NULL)",
    )
    op.create_check_constraint(
        "ck_ml_refinement_run_budgets",
        "ml_refinement_runs",
        "maximum_trials > 0 AND maximum_runtime_seconds > 0 "
        "AND combinations_planned >= 0 AND evaluations_computed >= 0 "
        "AND evaluations_reused >= 0 AND evaluations_omitted >= 0",
    )
    for name in (
        "cancellation_requested_at",
        "heartbeat_at",
        "claimed_at",
        "worker_id",
        "evaluations_selected",
    ):
        op.drop_column("ml_refinement_runs", name)
