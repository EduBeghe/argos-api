from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base


class PaperDeploymentRecord(Base):
    __tablename__ = "paper_deployments"
    __table_args__ = (
        CheckConstraint(
            "lifecycle IN ('active','paused','archived')",
            name="ck_paper_deployment_lifecycle",
        ),
        CheckConstraint(
            "strategy_kind IN ('baseline','ml_fixed_horizon_long_only')",
            name="ck_paper_deployment_strategy_kind",
        ),
        CheckConstraint(
            "length(btrim(display_name)) BETWEEN 1 AND 80",
            name="ck_paper_deployment_name",
        ),
        CheckConstraint(
            "(lifecycle = 'archived' AND archived_at IS NOT NULL "
            "AND is_primary = false) OR "
            "(lifecycle <> 'archived' AND archived_at IS NULL)",
            name="ck_paper_deployment_archive",
        ),
        Index(
            "uq_paper_deployment_name_live",
            text("lower(display_name)"),
            unique=True,
            postgresql_where=text("lifecycle <> 'archived'"),
        ),
        Index(
            "uq_paper_deployment_primary",
            "is_primary",
            unique=True,
            postgresql_where=text("is_primary"),
        ),
        Index(
            "uq_paper_deployment_strategy_live",
            "strategy_specification_id",
            unique=True,
            postgresql_where=text(
                "strategy_specification_id IS NOT NULL AND lifecycle <> 'archived'"
            ),
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    lifecycle: Mapped[str] = mapped_column(String(16), nullable=False)
    strategy_kind: Mapped[str] = mapped_column(String(48), nullable=False)
    strategy_specification_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_strategy_specifications.id", ondelete="RESTRICT"),
    )
    current_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("paper_trading_runs.id", ondelete="RESTRICT"),
    )
    is_primary: Mapped[bool] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
