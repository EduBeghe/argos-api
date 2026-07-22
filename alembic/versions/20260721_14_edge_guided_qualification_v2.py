"""Add edge-guided candidate qualification contract v2 evidence.

Revision ID: 20260721_14
Revises: 20260721_13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20260721_14"
down_revision: str | None = "20260721_13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "edge_guided_evaluation_runs",
        sa.Column(
            "qualification_contract_version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )
    op.add_column(
        "edge_guided_candidate_evaluations",
        sa.Column(
            "rejection_reasons",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "edge_guided_window_evidence",
        sa.Column(
            "rejection_reasons",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("edge_guided_window_evidence", "rejection_reasons")
    op.drop_column("edge_guided_candidate_evaluations", "rejection_reasons")
    op.drop_column("edge_guided_evaluation_runs", "qualification_contract_version")
