"""Index Decision Snapshot history reads.

Revision ID: 20260719_09
Revises: 20260718_08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260719_09"
down_revision: str | None = "20260718_08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_decisions_identity_timestamp",
        "decision_snapshots",
        [
            "asset",
            "provider",
            "market_type",
            "interval",
            "decision_timestamp",
            "id",
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_decisions_identity_timestamp", table_name="decision_snapshots")
