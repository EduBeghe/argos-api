from datetime import datetime, timedelta
from decimal import Decimal
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.decisions.persistence import DecisionSnapshotRecord
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.protected.models import (
    MLModelArtifactRecord,
    MLStrategySpecificationRecord,
)
from app.paper.deployment_models import PaperDeploymentRecord
from app.strategies.baseline_long_only import StrategyAction


class MLDecisionSnapshot(BaseModel):
    """Immutable evidence for one ML deployment decision."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    deployment_id: UUID
    strategy_specification_id: UUID
    model_artifact_id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    decision_timestamp: datetime
    source_candle_open_time: datetime
    source_market_price: Decimal
    signed_prediction: Decimal
    entry_threshold: Decimal
    model_artifact_sha256: str
    feature_schema: tuple[str, ...]
    feature_vector_sha256: str
    action: StrategyAction
    requested_target_portfolio_percentage: Decimal | None
    requested_position_percentage: Decimal | None
    applied_rule: str
    explanation: str
    position_is_open: bool
    scheduled_exit_at: datetime | None = None
    delayed_after_pause: bool = False
    execution_delay_seconds: int = 0

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        for timestamp in (
            self.decision_timestamp,
            self.source_candle_open_time,
            self.scheduled_exit_at,
        ):
            if timestamp is not None and (
                timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0)
            ):
                raise ValueError("timestamps must be timezone-aware UTC")
        if not self.source_market_price.is_finite() or self.source_market_price <= 0:
            raise ValueError("source market price must be finite and positive")
        if not self.signed_prediction.is_finite() or not Decimal(
            "-1"
        ) <= self.signed_prediction <= Decimal("1"):
            raise ValueError("signed prediction must be finite and between -1 and 1")
        if not self.entry_threshold.is_finite() or not Decimal(
            "-1"
        ) <= self.entry_threshold <= Decimal("1"):
            raise ValueError("entry threshold must be finite and between -1 and 1")
        if (
            len(self.model_artifact_sha256) != 64
            or len(self.feature_vector_sha256) != 64
        ):
            raise ValueError("artifact and feature-vector SHA-256 values are required")
        if not self.feature_schema or any(
            not name.strip() for name in self.feature_schema
        ):
            raise ValueError("feature schema must be non-empty")
        if not self.applied_rule.strip() or not self.explanation.strip():
            raise ValueError("rule and explanation must be non-empty")
        if self.execution_delay_seconds < 0:
            raise ValueError("execution delay must not be negative")
        if self.action is StrategyAction.OPEN_POSITION:
            valid = (
                not self.position_is_open
                and self.signed_prediction >= self.entry_threshold
                and self.requested_target_portfolio_percentage == Decimal("100")
                and self.requested_position_percentage is None
                and self.applied_rule == "ml_entry_threshold_met"
                and self.scheduled_exit_at is None
            )
        elif self.action is StrategyAction.SELL_ALL:
            valid = (
                self.position_is_open
                and self.requested_target_portfolio_percentage is None
                and self.requested_position_percentage == Decimal("100")
                and self.applied_rule == "ml_fixed_horizon_elapsed"
                and self.scheduled_exit_at is not None
                and self.decision_timestamp >= self.scheduled_exit_at
            )
        elif self.action is StrategyAction.NO_ACTION:
            valid = (
                self.requested_target_portfolio_percentage is None
                and self.requested_position_percentage is None
                and self.applied_rule
                in {"ml_entry_threshold_not_met", "ml_open_position_retained"}
            )
        else:
            valid = False
        if not valid:
            raise ValueError(
                "ML action, rule, position state, and sizing are inconsistent"
            )
        return self


class MLDecisionSnapshotRecord(Base):
    __tablename__ = "ml_decision_snapshots"
    __table_args__ = (
        CheckConstraint(
            "action IN ('open_position','sell_all','no_action')",
            name="ck_ml_decision_action",
        ),
        CheckConstraint(
            "source_market_price > 0 AND signed_prediction BETWEEN -1 AND 1 "
            "AND entry_threshold BETWEEN -1 AND 1",
            name="ck_ml_decision_values",
        ),
        CheckConstraint(
            "length(model_artifact_sha256)=64 AND length(feature_vector_sha256)=64 "
            "AND length(btrim(applied_rule))>0 AND length(btrim(explanation))>0",
            name="ck_ml_decision_evidence",
        ),
        CheckConstraint("execution_delay_seconds >= 0", name="ck_ml_decision_delay"),
        CheckConstraint(
            "(action='open_position' AND position_is_open=false AND "
            "signed_prediction >= entry_threshold AND "
            "requested_target_portfolio_percentage=100 AND "
            "requested_position_percentage IS NULL AND "
            "applied_rule='ml_entry_threshold_met' AND scheduled_exit_at IS NULL) OR "
            "(action='sell_all' AND position_is_open=true AND "
            "requested_target_portfolio_percentage IS NULL AND "
            "requested_position_percentage=100 AND "
            "applied_rule='ml_fixed_horizon_elapsed' AND "
            "scheduled_exit_at IS NOT NULL AND "
            "decision_timestamp >= scheduled_exit_at) OR "
            "(action='no_action' AND requested_target_portfolio_percentage IS NULL AND "
            "requested_position_percentage IS NULL AND applied_rule IN "
            "('ml_entry_threshold_not_met','ml_open_position_retained'))",
            name="ck_ml_decision_action_evidence",
        ),
        UniqueConstraint(
            "deployment_id",
            "asset",
            "provider",
            "market_type",
            "interval",
            "decision_timestamp",
            name="uq_ml_decision_deployment_boundary",
        ),
        Index(
            "ix_ml_decision_identity_time",
            "deployment_id",
            "asset",
            "provider",
            "market_type",
            "interval",
            "decision_timestamp",
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    deployment_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(PaperDeploymentRecord.id),
        nullable=False,
    )
    strategy_specification_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(MLStrategySpecificationRecord.id),
        nullable=False,
    )
    model_artifact_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey(MLModelArtifactRecord.id),
        nullable=False,
    )
    asset: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    decision_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_candle_open_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_market_price: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    signed_prediction: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    entry_threshold: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    model_artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_schema: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    feature_vector_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_target_portfolio_percentage: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    requested_position_percentage: Mapped[Decimal | None] = mapped_column(
        Numeric(38, 18)
    )
    applied_rule: Mapped[str] = mapped_column(String(64), nullable=False)
    explanation: Mapped[str] = mapped_column(String(512), nullable=False)
    position_is_open: Mapped[bool] = mapped_column(Boolean, nullable=False)
    scheduled_exit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delayed_after_pause: Mapped[bool] = mapped_column(Boolean, nullable=False)
    execution_delay_seconds: Mapped[int] = mapped_column(Integer, nullable=False)


class PaperDecisionIdentityRecord(Base):
    __tablename__ = "paper_decision_identities"
    __table_args__ = (
        CheckConstraint(
            "(kind='baseline' AND baseline_snapshot_id IS NOT NULL AND "
            "ml_snapshot_id IS NULL) OR "
            "(kind='ml' AND baseline_snapshot_id IS NULL AND "
            "ml_snapshot_id IS NOT NULL)",
            name="ck_paper_decision_identity_xor",
        ),
        UniqueConstraint(
            "baseline_snapshot_id", name="uq_paper_decision_identity_baseline"
        ),
        UniqueConstraint("ml_snapshot_id", name="uq_paper_decision_identity_ml"),
        UniqueConstraint(
            "id",
            "baseline_snapshot_id",
            name="uq_paper_decision_identity_baseline_pair",
        ),
        UniqueConstraint(
            "id", "ml_snapshot_id", name="uq_paper_decision_identity_ml_pair"
        ),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    baseline_snapshot_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey(DecisionSnapshotRecord.id)
    )
    ml_snapshot_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey(MLDecisionSnapshotRecord.id)
    )


async def persist_ml_decision(
    session: AsyncSession, snapshot: MLDecisionSnapshot
) -> PaperDecisionIdentityRecord:
    existing = await session.scalar(
        select(PaperDecisionIdentityRecord)
        .join(
            MLDecisionSnapshotRecord,
            MLDecisionSnapshotRecord.id == PaperDecisionIdentityRecord.ml_snapshot_id,
        )
        .where(MLDecisionSnapshotRecord.id == snapshot.id)
    )
    if existing is not None:
        return existing
    record = MLDecisionSnapshotRecord(**snapshot.model_dump(mode="python"))
    session.add(record)
    identity = PaperDecisionIdentityRecord(
        id=uuid4(), kind="ml", baseline_snapshot_id=None, ml_snapshot_id=record.id
    )
    session.add(identity)
    await session.flush()
    return identity


async def persist_baseline_decision_identity(
    session: AsyncSession, snapshot_id: UUID
) -> PaperDecisionIdentityRecord:
    existing = await session.scalar(
        select(PaperDecisionIdentityRecord).where(
            PaperDecisionIdentityRecord.baseline_snapshot_id == snapshot_id
        )
    )
    if existing is not None:
        return existing
    identity = PaperDecisionIdentityRecord(
        id=uuid4(),
        kind="baseline",
        baseline_snapshot_id=snapshot_id,
        ml_snapshot_id=None,
    )
    session.add(identity)
    await session.flush()
    return identity
