"""Add weekly Strategy discovery contract fields.

Revision ID: 20260722_15
Revises: 20260721_14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260722_15"
down_revision: str | None = "20260721_14"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "edge_guided_stage_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("candidate_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("window_ordinal", sa.Integer(), nullable=False),
        sa.Column("segment_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("segment_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["edge_guided_evaluation_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "cache_key", name="uq_edge_guided_stage_run_key"),
    )
    op.create_index(
        "ix_edge_guided_stage_cache_key",
        "edge_guided_stage_evidence",
        ["cache_key"],
    )
    op.create_index(
        "ix_edge_guided_stage_candidate",
        "edge_guided_stage_evidence",
        ["candidate_fingerprint", "stage"],
    )
    _add_column_if_missing(
        "edge_guided_evaluation_runs",
        sa.Column("reuse_key", sa.String(length=64), nullable=True),
    )
    _add_column_if_missing(
        "edge_guided_window_evidence",
        sa.Column("cache_key", sa.String(length=64), nullable=True),
    )
    _add_column_if_missing(
        "edge_guided_evaluation_runs",
        sa.Column(
            "dataset_revision",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    _add_column_if_missing(
        "edge_guided_evaluation_runs",
        sa.Column("research_outcome", sa.String(length=64), nullable=True),
    )
    _add_column_if_missing(
        "edge_guided_evaluation_runs",
        sa.Column("outcome_explanation", sa.Text(), nullable=True),
    )
    _add_column_if_missing(
        "edge_guided_evaluation_runs",
        sa.Column("funnel", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    _add_column_if_missing(
        "edge_guided_evaluation_runs",
        sa.Column("timings", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    indexes = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_indexes("edge_guided_evaluation_runs")
    }
    if "uq_edge_guided_runs_reuse_key" not in indexes:
        op.create_index(
            "uq_edge_guided_runs_reuse_key",
            "edge_guided_evaluation_runs",
            ["reuse_key"],
            unique=True,
        )
    window_indexes = {
        item["name"]
        for item in sa.inspect(op.get_bind()).get_indexes("edge_guided_window_evidence")
    }
    if "ix_edge_guided_windows_cache_key" not in window_indexes:
        op.create_index(
            "ix_edge_guided_windows_cache_key",
            "edge_guided_window_evidence",
            ["cache_key"],
        )
    op.alter_column(
        "edge_guided_candidate_evaluations",
        "test_metrics",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=True,
    )


def downgrade() -> None:
    op.execute(
        "UPDATE edge_guided_candidate_evaluations "
        "SET test_metrics = '{}'::jsonb WHERE test_metrics IS NULL"
    )
    op.alter_column(
        "edge_guided_candidate_evaluations",
        "test_metrics",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
    )
    op.drop_index(
        "uq_edge_guided_runs_reuse_key",
        table_name="edge_guided_evaluation_runs",
    )
    op.drop_index(
        "ix_edge_guided_windows_cache_key",
        table_name="edge_guided_window_evidence",
    )
    op.drop_column("edge_guided_window_evidence", "cache_key")
    op.drop_index(
        "ix_edge_guided_stage_candidate",
        table_name="edge_guided_stage_evidence",
    )
    op.drop_index(
        "ix_edge_guided_stage_cache_key",
        table_name="edge_guided_stage_evidence",
    )
    op.drop_table("edge_guided_stage_evidence")
    for name in (
        "timings",
        "funnel",
        "outcome_explanation",
        "research_outcome",
        "dataset_revision",
        "reuse_key",
    ):
        op.drop_column("edge_guided_evaluation_runs", name)


def _add_column_if_missing(table_name: str, column: sa.Column[object]) -> None:
    columns = {
        item["name"] for item in sa.inspect(op.get_bind()).get_columns(table_name)
    }
    if column.name not in columns:
        op.add_column(table_name, column)
