# ruff: noqa: E501
"""Add protected ML evidence and isolated Paper deployments.

Revision ID: 20260804_26
Revises: 20260731_25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260804_26"
down_revision: str | None = "20260731_25"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ml_temporal_manifests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "evaluation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_refinement_evaluations.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("source", sa.String(40), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("manifest_version", sa.Integer(), nullable=False),
        sa.Column("split_policy_version", sa.Integer(), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(64), nullable=False),
        sa.Column("manifest_fingerprint", sa.String(64)),
        sa.Column("manifest", postgresql.JSONB()),
        sa.Column("fingerprint_verified", sa.Boolean()),
        sa.Column("counts_verified", sa.Boolean()),
        sa.Column("worker_id", sa.String(128)),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("cancellation_requested_at", sa.DateTime(timezone=True)),
        sa.Column("failure_code", sa.String(64)),
        sa.Column("failure_message", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "source IN ('persisted_at_execution','reconstructed_from_contract')",
            name="ck_ml_temporal_manifest_source",
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_temporal_manifest_status",
        ),
        sa.UniqueConstraint(
            "evaluation_id",
            "manifest_version",
            "dataset_fingerprint",
            name="uq_ml_temporal_manifest_identity",
        ),
    )
    op.create_table(
        "ml_final_validations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_key", sa.String(64), nullable=False, unique=True),
        sa.Column(
            "campaign_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaigns.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "family_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_search_campaign_families.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "evaluation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_refinement_evaluations.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "temporal_manifest_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_temporal_manifests.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("classification", sa.String(48)),
        sa.Column("frozen_inputs", postgresql.JSONB(), nullable=False),
        sa.Column("execution_manifest", postgresql.JSONB()),
        sa.Column("scenario_metrics", postgresql.JSONB()),
        sa.Column("failed_rules", postgresql.JSONB()),
        sa.Column("protected_data_read_at", sa.DateTime(timezone=True)),
        sa.Column("worker_id", sa.String(128)),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("cancellation_requested_at", sa.DateTime(timezone=True)),
        sa.Column("failure_code", sa.String(64)),
        sa.Column("failure_message", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_final_validation_status",
        ),
    )
    op.create_table(
        "ml_final_validation_trades",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "validation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_final_validations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scenario", sa.String(16), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("entry_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.Numeric(38, 18), nullable=False),
        sa.Column("exit_price", sa.Numeric(38, 18), nullable=False),
        sa.Column("quantity", sa.Numeric(38, 18), nullable=False),
        sa.Column("fees", sa.Numeric(38, 18), nullable=False),
        sa.Column("slippage", sa.Numeric(38, 18), nullable=False),
        sa.Column("net_return", sa.Numeric(38, 18), nullable=False),
        sa.Column("exit_reason", sa.String(48), nullable=False),
        sa.UniqueConstraint(
            "validation_id", "scenario", "ordinal", name="uq_ml_final_validation_trade"
        ),
    )
    op.create_table(
        "ml_strategy_specifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "final_validation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_final_validations.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("strategy_version", sa.Integer(), nullable=False),
        sa.Column("specification", postgresql.JSONB(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "ml_model_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "strategy_specification_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_strategy_specifications.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("trusted_provenance", sa.String(64), nullable=False),
        sa.Column("format_version", sa.Integer(), nullable=False),
        sa.Column("compression", sa.String(16), nullable=False),
        sa.Column("byte_length", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("engine", sa.String(16), nullable=False),
        sa.Column("library_versions", postgresql.JSONB(), nullable=False),
        sa.Column("feature_schema", postgresql.JSONB(), nullable=False),
        sa.Column("training_identity", postgresql.JSONB(), nullable=False),
        sa.Column("artifact", sa.LargeBinary(), nullable=False),
        sa.Column("load_validated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "paper_deployments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("display_name", sa.String(80), nullable=False),
        sa.Column("lifecycle", sa.String(16), nullable=False),
        sa.Column("strategy_kind", sa.String(48), nullable=False),
        sa.Column(
            "strategy_specification_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_strategy_specifications.id", ondelete="RESTRICT"),
        ),
        sa.Column("current_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("is_primary", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("paused_at", sa.DateTime(timezone=True)),
        sa.Column("archived_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "lifecycle IN ('active','paused','archived')",
            name="ck_paper_deployment_lifecycle",
        ),
        sa.CheckConstraint(
            "strategy_kind IN ('baseline','ml_fixed_horizon_long_only')",
            name="ck_paper_deployment_strategy_kind",
        ),
        sa.CheckConstraint(
            "length(btrim(display_name)) BETWEEN 1 AND 80",
            name="ck_paper_deployment_name",
        ),
        sa.CheckConstraint(
            "(lifecycle = 'archived' AND archived_at IS NOT NULL "
            "AND is_primary = false) OR "
            "(lifecycle <> 'archived' AND archived_at IS NULL)",
            name="ck_paper_deployment_archive",
        ),
    )
    op.add_column(
        "paper_trading_runs", sa.Column("deployment_id", postgresql.UUID(as_uuid=True))
    )
    op.execute("DROP INDEX IF EXISTS uq_paper_runs_active_currency")
    op.execute(
        """
    DO $$ DECLARE root record; deployment uuid; active_root uuid; roots integer;
    BEGIN
      IF EXISTS (SELECT 1 FROM paper_trading_runs r WHERE predecessor_run_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM paper_trading_runs p WHERE p.id=r.predecessor_run_id)) THEN RAISE EXCEPTION 'paper predecessor corruption'; END IF;
      IF (SELECT count(*) FROM paper_trading_runs WHERE status='active') > 1 THEN RAISE EXCEPTION 'multiple active paper runs'; END IF;
      FOR root IN SELECT * FROM paper_trading_runs WHERE predecessor_run_id IS NULL ORDER BY started_at,id LOOP
        deployment := gen_random_uuid();
        INSERT INTO paper_deployments(id,display_name,lifecycle,strategy_kind,current_run_id,is_primary,created_at,archived_at)
        VALUES(deployment, 'Baseline ' || left(root.id::text,8), 'archived', 'baseline', NULL, false, root.started_at, root.started_at);
        WITH RECURSIVE lineage AS (SELECT id FROM paper_trading_runs WHERE id=root.id UNION ALL SELECT r.id FROM paper_trading_runs r JOIN lineage l ON r.predecessor_run_id=l.id)
        UPDATE paper_trading_runs SET deployment_id=deployment WHERE id IN (SELECT id FROM lineage);
        UPDATE paper_deployments SET current_run_id=(SELECT id FROM paper_trading_runs WHERE deployment_id=deployment ORDER BY started_at DESC,id DESC LIMIT 1) WHERE id=deployment;
        IF EXISTS(SELECT 1 FROM paper_trading_runs WHERE deployment_id=deployment AND status='active') THEN
          UPDATE paper_deployments SET lifecycle='active', archived_at=NULL WHERE id=deployment;
        END IF;
      END LOOP;
      IF EXISTS(SELECT 1 FROM paper_trading_runs WHERE deployment_id IS NULL) THEN
        RAISE EXCEPTION 'paper predecessor cycle or ambiguous lineage';
      END IF;
      UPDATE paper_deployments SET is_primary=true WHERE id=(SELECT deployment_id FROM paper_trading_runs WHERE status='active' LIMIT 1);
    END $$;
    """
    )
    op.alter_column("paper_trading_runs", "deployment_id", nullable=False)
    op.create_foreign_key(
        "fk_paper_runs_deployment",
        "paper_trading_runs",
        "paper_deployments",
        ["deployment_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_paper_deployment_current_run",
        "paper_deployments",
        "paper_trading_runs",
        ["current_run_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "uq_paper_runs_active_deployment",
        "paper_trading_runs",
        ["deployment_id"],
        unique=True,
        postgresql_where=sa.text("status='active'"),
    )
    op.create_index(
        "uq_paper_deployment_primary",
        "paper_deployments",
        ["is_primary"],
        unique=True,
        postgresql_where=sa.text("is_primary"),
    )
    op.create_index(
        "uq_paper_deployment_name_live",
        "paper_deployments",
        [sa.text("lower(display_name)")],
        unique=True,
        postgresql_where=sa.text("lifecycle <> 'archived'"),
    )
    op.create_index(
        "uq_paper_deployment_strategy_live",
        "paper_deployments",
        ["strategy_specification_id"],
        unique=True,
        postgresql_where=sa.text(
            "strategy_specification_id IS NOT NULL AND lifecycle <> 'archived'"
        ),
    )
    op.create_table(
        "ml_decision_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "deployment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("paper_deployments.id"),
            nullable=False,
        ),
        sa.Column(
            "strategy_specification_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_strategy_specifications.id"),
            nullable=False,
        ),
        sa.Column(
            "model_artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_model_artifacts.id"),
            nullable=False,
        ),
        sa.Column("asset", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("interval", sa.String(16), nullable=False),
        sa.Column("decision_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "source_candle_open_time", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("source_market_price", sa.Numeric(38, 18), nullable=False),
        sa.Column("signed_prediction", sa.Numeric(38, 18), nullable=False),
        sa.Column("entry_threshold", sa.Numeric(38, 18), nullable=False),
        sa.Column("model_artifact_sha256", sa.String(64), nullable=False),
        sa.Column("feature_schema", postgresql.JSONB(), nullable=False),
        sa.Column("feature_vector_sha256", sa.String(64), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("requested_target_portfolio_percentage", sa.Numeric(38, 18)),
        sa.Column("requested_position_percentage", sa.Numeric(38, 18)),
        sa.Column("applied_rule", sa.String(64), nullable=False),
        sa.Column("explanation", sa.String(512), nullable=False),
        sa.Column("position_is_open", sa.Boolean(), nullable=False),
        sa.Column("scheduled_exit_at", sa.DateTime(timezone=True)),
        sa.Column("delayed_after_pause", sa.Boolean(), nullable=False),
        sa.Column("execution_delay_seconds", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "action IN ('open_position','sell_all','no_action')",
            name="ck_ml_decision_action",
        ),
        sa.CheckConstraint(
            "source_market_price > 0 AND signed_prediction BETWEEN -1 AND 1 AND entry_threshold BETWEEN -1 AND 1",
            name="ck_ml_decision_values",
        ),
        sa.CheckConstraint(
            "length(model_artifact_sha256)=64 AND length(feature_vector_sha256)=64 AND length(btrim(applied_rule))>0 AND length(btrim(explanation))>0",
            name="ck_ml_decision_evidence",
        ),
        sa.CheckConstraint("execution_delay_seconds >= 0", name="ck_ml_decision_delay"),
        sa.CheckConstraint(
            "(action='open_position' AND position_is_open=false AND signed_prediction >= entry_threshold AND requested_target_portfolio_percentage=100 AND requested_position_percentage IS NULL AND applied_rule='ml_entry_threshold_met' AND scheduled_exit_at IS NULL) OR (action='sell_all' AND position_is_open=true AND requested_target_portfolio_percentage IS NULL AND requested_position_percentage=100 AND applied_rule='ml_fixed_horizon_elapsed' AND scheduled_exit_at IS NOT NULL AND decision_timestamp >= scheduled_exit_at) OR (action='no_action' AND requested_target_portfolio_percentage IS NULL AND requested_position_percentage IS NULL AND applied_rule IN ('ml_entry_threshold_not_met','ml_open_position_retained'))",
            name="ck_ml_decision_action_evidence",
        ),
        sa.UniqueConstraint(
            "deployment_id",
            "asset",
            "provider",
            "market_type",
            "interval",
            "decision_timestamp",
            name="uq_ml_decision_deployment_boundary",
        ),
    )
    op.create_index(
        "ix_ml_decision_identity_time",
        "ml_decision_snapshots",
        [
            "deployment_id",
            "asset",
            "provider",
            "market_type",
            "interval",
            "decision_timestamp",
        ],
    )
    op.create_table(
        "paper_decision_identities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column(
            "baseline_snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("decision_snapshots.id"),
        ),
        sa.Column(
            "ml_snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ml_decision_snapshots.id"),
        ),
        sa.CheckConstraint(
            "(kind='baseline' AND baseline_snapshot_id IS NOT NULL AND ml_snapshot_id IS NULL) OR (kind='ml' AND baseline_snapshot_id IS NULL AND ml_snapshot_id IS NOT NULL)",
            name="ck_paper_decision_identity_xor",
        ),
        sa.UniqueConstraint(
            "baseline_snapshot_id", name="uq_paper_decision_identity_baseline"
        ),
        sa.UniqueConstraint("ml_snapshot_id", name="uq_paper_decision_identity_ml"),
        sa.UniqueConstraint(
            "id",
            "baseline_snapshot_id",
            name="uq_paper_decision_identity_baseline_pair",
        ),
        sa.UniqueConstraint(
            "id", "ml_snapshot_id", name="uq_paper_decision_identity_ml_pair"
        ),
    )
    op.add_column(
        "paper_decision_cycles",
        sa.Column("ml_decision_snapshot_id", postgresql.UUID(as_uuid=True)),
    )
    op.add_column(
        "paper_decision_cycles",
        sa.Column("decision_identity_id", postgresql.UUID(as_uuid=True)),
    )
    op.add_column("paper_decision_cycles", sa.Column("decision_kind", sa.String(16)))
    op.add_column(
        "paper_trades", sa.Column("decision_identity_id", postgresql.UUID(as_uuid=True))
    )
    op.execute(
        """
        INSERT INTO paper_decision_identities(id, kind, baseline_snapshot_id)
        SELECT gen_random_uuid(), 'baseline', decision_snapshot_id
        FROM (
          SELECT decision_snapshot_id FROM paper_decision_cycles WHERE decision_snapshot_id IS NOT NULL
          UNION
          SELECT decision_snapshot_id FROM paper_trades WHERE decision_snapshot_id IS NOT NULL
        ) evidence;
        UPDATE paper_decision_cycles c SET decision_identity_id=i.id, decision_kind='baseline'
        FROM paper_decision_identities i WHERE i.baseline_snapshot_id=c.decision_snapshot_id;
        UPDATE paper_trades t SET decision_identity_id=i.id
        FROM paper_decision_identities i WHERE i.baseline_snapshot_id=t.decision_snapshot_id;
        """
    )
    op.create_foreign_key(
        "fk_paper_cycles_ml_decision",
        "paper_decision_cycles",
        "ml_decision_snapshots",
        ["ml_decision_snapshot_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_paper_cycles_decision_identity",
        "paper_decision_cycles",
        "paper_decision_identities",
        ["decision_identity_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_paper_cycles_baseline_identity_pair",
        "paper_decision_cycles",
        "paper_decision_identities",
        ["decision_identity_id", "decision_snapshot_id"],
        ["id", "baseline_snapshot_id"],
    )
    op.create_foreign_key(
        "fk_paper_cycles_ml_identity_pair",
        "paper_decision_cycles",
        "paper_decision_identities",
        ["decision_identity_id", "ml_decision_snapshot_id"],
        ["id", "ml_snapshot_id"],
    )
    op.create_foreign_key(
        "fk_paper_trades_decision_identity",
        "paper_trades",
        "paper_decision_identities",
        ["decision_identity_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_paper_trades_baseline_identity_pair",
        "paper_trades",
        "paper_decision_identities",
        ["decision_identity_id", "decision_snapshot_id"],
        ["id", "baseline_snapshot_id"],
    )
    op.alter_column("paper_trades", "decision_identity_id", nullable=False)
    op.alter_column("paper_trades", "decision_snapshot_id", nullable=True)
    op.drop_constraint(
        "ck_paper_cycles_lifecycle", "paper_decision_cycles", type_="check"
    )
    op.create_check_constraint(
        "ck_paper_cycles_lifecycle",
        "paper_decision_cycles",
        "(status = 'processing' AND completed_at IS NULL AND result IS NULL AND source_score_id IS NULL AND decision_snapshot_id IS NULL AND ml_decision_snapshot_id IS NULL AND decision_identity_id IS NULL AND decision_kind IS NULL AND paper_trade_id IS NULL) OR (status = 'completed' AND completed_at IS NOT NULL AND result IS NOT NULL AND decision_identity_id IS NOT NULL AND ((decision_kind = 'baseline' AND source_score_id IS NOT NULL AND decision_snapshot_id IS NOT NULL AND ml_decision_snapshot_id IS NULL) OR (decision_kind = 'ml' AND source_score_id IS NULL AND decision_snapshot_id IS NULL AND ml_decision_snapshot_id IS NOT NULL)) AND ((result = 'no_action' AND paper_trade_id IS NULL) OR (result IN ('open_position','sell_all') AND paper_trade_id IS NOT NULL)))",
    )
    op.create_unique_constraint(
        "uq_paper_cycles_ml_decision_snapshot",
        "paper_decision_cycles",
        ["ml_decision_snapshot_id"],
    )
    op.create_unique_constraint(
        "uq_paper_cycles_decision_identity",
        "paper_decision_cycles",
        ["decision_identity_id"],
    )
    op.execute("DROP INDEX IF EXISTS uq_paper_trades_decision")
    op.create_index(
        "uq_paper_trades_baseline_decision",
        "paper_trades",
        ["decision_snapshot_id"],
        unique=True,
        postgresql_where=sa.text("decision_snapshot_id IS NOT NULL"),
    )
    op.create_index(
        "uq_paper_trades_decision_identity",
        "paper_trades",
        ["decision_identity_id"],
        unique=True,
    )


def downgrade() -> None:
    connection = op.get_bind()
    if connection.execute(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM ml_final_validations) OR "
            "EXISTS(SELECT 1 FROM paper_deployments "
            "WHERE strategy_kind <> 'baseline')"
        )
    ).scalar():
        raise RuntimeError(
            "downgrade refused: archive/export protected ML and deployment data first"
        )
    op.drop_index("uq_paper_trades_decision_identity", table_name="paper_trades")
    op.drop_index("uq_paper_trades_baseline_decision", table_name="paper_trades")
    op.create_index(
        "uq_paper_trades_decision",
        "paper_trades",
        ["decision_snapshot_id"],
        unique=True,
    )
    op.drop_constraint(
        "ck_paper_cycles_lifecycle", "paper_decision_cycles", type_="check"
    )
    op.create_check_constraint(
        "ck_paper_cycles_lifecycle",
        "paper_decision_cycles",
        "(status = 'processing' AND completed_at IS NULL AND result IS NULL AND source_score_id IS NULL AND decision_snapshot_id IS NULL AND paper_trade_id IS NULL) OR (status = 'completed' AND completed_at IS NOT NULL AND result IS NOT NULL AND source_score_id IS NOT NULL AND decision_snapshot_id IS NOT NULL AND ((result = 'no_action' AND paper_trade_id IS NULL) OR (result IN ('open_position','sell_all') AND paper_trade_id IS NOT NULL)))",
    )
    op.drop_constraint(
        "uq_paper_cycles_decision_identity", "paper_decision_cycles", type_="unique"
    )
    op.drop_constraint(
        "uq_paper_cycles_ml_decision_snapshot", "paper_decision_cycles", type_="unique"
    )
    op.drop_constraint(
        "fk_paper_trades_baseline_identity_pair", "paper_trades", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_paper_trades_decision_identity", "paper_trades", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_paper_cycles_ml_identity_pair", "paper_decision_cycles", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_paper_cycles_baseline_identity_pair",
        "paper_decision_cycles",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_paper_cycles_decision_identity", "paper_decision_cycles", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_paper_cycles_ml_decision", "paper_decision_cycles", type_="foreignkey"
    )
    op.alter_column("paper_trades", "decision_snapshot_id", nullable=False)
    op.drop_column("paper_trades", "decision_identity_id")
    op.drop_column("paper_decision_cycles", "decision_kind")
    op.drop_column("paper_decision_cycles", "decision_identity_id")
    op.drop_column("paper_decision_cycles", "ml_decision_snapshot_id")
    op.drop_table("paper_decision_identities")
    op.drop_index("ix_ml_decision_identity_time", table_name="ml_decision_snapshots")
    op.drop_table("ml_decision_snapshots")
    op.drop_constraint(
        "fk_paper_deployment_current_run", "paper_deployments", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_paper_runs_deployment", "paper_trading_runs", type_="foreignkey"
    )
    op.drop_column("paper_trading_runs", "deployment_id")
    op.create_index(
        "uq_paper_runs_active_currency",
        "paper_trading_runs",
        ["account_currency"],
        unique=True,
        postgresql_where=sa.text("status='active'"),
    )
    for table in (
        "paper_deployments",
        "ml_model_artifacts",
        "ml_strategy_specifications",
        "ml_final_validation_trades",
        "ml_final_validations",
        "ml_temporal_manifests",
    ):
        op.drop_table(table)
