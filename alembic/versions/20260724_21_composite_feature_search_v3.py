"""Add Composite Feature Search v3 evidence.

Revision ID: 20260724_21
Revises: 20260724_20
"""

import sqlalchemy as sa

from alembic import op

revision = "20260724_21"
down_revision = "20260724_20"
branch_labels = None
depends_on = None


_RUN_COUNTER_COLUMNS = (
    "candidate_simulations_computed",
    "candidate_simulations_reused",
    "production_simulations_computed",
    "production_simulations_reused",
    "buy_and_hold_simulations_computed",
    "buy_and_hold_simulations_reused",
    "normalizations_computed",
    "normalizations_reused",
    "simulations_avoided",
    "row_slices_computed",
    "row_slices_reused",
    "threshold_fits_computed",
    "threshold_fits_reused",
)

_RUN_BOOLEAN_COLUMNS = (
    "persistence_truncated",
    "depth_quota_reached",
    "global_compute_budget_exhausted",
)


def upgrade() -> None:
    run_columns = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_columns("composite_search_runs")
    }
    candidate_columns = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_columns("composite_search_candidates")
    }
    if "best_economic_candidate_fingerprint" not in run_columns:
        op.add_column(
            "composite_search_runs",
            sa.Column("best_economic_candidate_fingerprint", sa.String(64)),
        )
    if "best_exploratory_candidate_fingerprint" not in run_columns:
        op.add_column(
            "composite_search_runs",
            sa.Column("best_exploratory_candidate_fingerprint", sa.String(64)),
        )
    for name in _RUN_COUNTER_COLUMNS:
        if name not in run_columns:
            op.add_column(
                "composite_search_runs",
                sa.Column(name, sa.Integer(), nullable=False, server_default="0"),
            )
            op.alter_column("composite_search_runs", name, server_default=None)
    for name in _RUN_BOOLEAN_COLUMNS:
        if name not in run_columns:
            op.add_column(
                "composite_search_runs",
                sa.Column(
                    name, sa.Boolean(), nullable=False, server_default=sa.false()
                ),
            )
            op.alter_column("composite_search_runs", name, server_default=None)

    if "selection_track" not in candidate_columns:
        op.add_column(
            "composite_search_candidates", sa.Column("selection_track", sa.String(16))
        )
    if "exploration_reason" not in candidate_columns:
        op.add_column(
            "composite_search_candidates",
            sa.Column("exploration_reason", sa.String(128)),
        )
    if "distance_to_training_qualification" not in candidate_columns:
        op.add_column(
            "composite_search_candidates",
            sa.Column("distance_to_training_qualification", sa.String(64)),
        )
    for name, size in (
        ("operational_search_stage", 32),
        ("parent_operational_configuration_fingerprint", 64),
        ("operational_refinement_reason", 128),
    ):
        if name not in candidate_columns:
            op.add_column(
                "composite_search_candidates", sa.Column(name, sa.String(size))
            )
    constraints = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_check_constraints(
            "composite_search_candidates"
        )
    }
    if "ck_composite_search_candidate_selection_track" not in constraints:
        op.create_check_constraint(
            "ck_composite_search_candidate_selection_track",
            "composite_search_candidates",
            "selection_track IS NULL OR selection_track IN ('economic', 'exploratory')",
        )
    if "ck_composite_search_exploratory_not_promoted" not in constraints:
        op.create_check_constraint(
            "ck_composite_search_exploratory_not_promoted",
            "composite_search_candidates",
            "selection_track IS NULL OR selection_track = 'economic' "
            "OR status = 'retained_training'",
        )
    if "ck_composite_search_operational_stage" not in constraints:
        op.create_check_constraint(
            "ck_composite_search_operational_stage",
            "composite_search_candidates",
            "operational_search_stage IS NULL OR operational_search_stage IN "
            "('coarse', 'local_refinement')",
        )


def downgrade() -> None:
    op.drop_constraint(
        "ck_composite_search_operational_stage",
        "composite_search_candidates",
        type_="check",
    )
    op.drop_constraint(
        "ck_composite_search_exploratory_not_promoted",
        "composite_search_candidates",
        type_="check",
    )
    op.drop_constraint(
        "ck_composite_search_candidate_selection_track",
        "composite_search_candidates",
        type_="check",
    )
    op.drop_column("composite_search_candidates", "distance_to_training_qualification")
    op.drop_column("composite_search_candidates", "exploration_reason")
    op.drop_column("composite_search_candidates", "selection_track")
    op.drop_column("composite_search_candidates", "operational_refinement_reason")
    op.drop_column(
        "composite_search_candidates",
        "parent_operational_configuration_fingerprint",
    )
    op.drop_column("composite_search_candidates", "operational_search_stage")
    for name in reversed(_RUN_BOOLEAN_COLUMNS):
        op.drop_column("composite_search_runs", name)
    for name in reversed(_RUN_COUNTER_COLUMNS):
        op.drop_column("composite_search_runs", name)
    op.drop_column("composite_search_runs", "best_exploratory_candidate_fingerprint")
    op.drop_column("composite_search_runs", "best_economic_candidate_fingerprint")
