from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base

METRIC = Numeric(38, 18)


class MLTemporalManifestRecord(Base):
    __tablename__ = "ml_temporal_manifests"
    __table_args__ = (
        UniqueConstraint(
            "evaluation_id",
            "manifest_version",
            "dataset_fingerprint",
            name="uq_ml_temporal_manifest_identity",
        ),
        CheckConstraint(
            "source IN ('persisted_at_execution','reconstructed_from_contract')",
            name="ck_ml_temporal_manifest_source",
        ),
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_temporal_manifest_status",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    evaluation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_refinement_evaluations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    manifest_version: Mapped[int] = mapped_column(Integer, nullable=False)
    split_policy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_fingerprint: Mapped[str | None] = mapped_column(String(64))
    manifest: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    fingerprint_verified: Mapped[bool | None] = mapped_column()
    counts_verified: Mapped[bool | None] = mapped_column()
    worker_id: Mapped[str | None] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_message: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MLFinalValidationRecord(Base):
    __tablename__ = "ml_final_validations"
    __table_args__ = (
        UniqueConstraint("experiment_key", name="uq_ml_final_validation_experiment"),
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="ck_ml_final_validation_status",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    campaign_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaigns.id", ondelete="RESTRICT"),
        nullable=False,
    )
    family_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_search_campaign_families.id", ondelete="RESTRICT"),
        nullable=False,
    )
    evaluation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_refinement_evaluations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    temporal_manifest_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_temporal_manifests.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    classification: Mapped[str | None] = mapped_column(String(48))
    frozen_inputs: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    execution_manifest: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    scenario_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    failed_rules: Mapped[list[str] | None] = mapped_column(JSONB)
    protected_data_read_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    worker_id: Mapped[str | None] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_message: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MLFinalValidationTradeRecord(Base):
    __tablename__ = "ml_final_validation_trades"
    __table_args__ = (
        UniqueConstraint(
            "validation_id", "scenario", "ordinal", name="uq_ml_final_validation_trade"
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    validation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_final_validations.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    exit_price: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    fees: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    slippage: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    net_return: Mapped[Decimal] = mapped_column(METRIC, nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(48), nullable=False)


class MLStrategySpecificationRecord(Base):
    __tablename__ = "ml_strategy_specifications"
    __table_args__ = (
        UniqueConstraint("final_validation_id", name="uq_ml_strategy_validation"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    final_validation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_final_validations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    specification: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MLModelArtifactRecord(Base):
    __tablename__ = "ml_model_artifacts"
    __table_args__ = (
        UniqueConstraint("strategy_specification_id", name="uq_ml_artifact_strategy"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    strategy_specification_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("ml_strategy_specifications.id", ondelete="RESTRICT"),
        nullable=False,
    )
    trusted_provenance: Mapped[str] = mapped_column(String(64), nullable=False)
    format_version: Mapped[int] = mapped_column(Integer, nullable=False)
    compression: Mapped[str] = mapped_column(String(16), nullable=False)
    byte_length: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    engine: Mapped[str] = mapped_column(String(16), nullable=False)
    library_versions: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False)
    feature_schema: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    training_identity: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    artifact: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    load_validated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
