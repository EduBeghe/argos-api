from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.optimization.targeted_refinement import (
    TARGETED_REFINEMENT_CONTRACT_VERSION,
    TargetedCandidateResult,
    TargetedRefinementResult,
)


class TargetedRefinementRunRecord(Base):
    __tablename__ = "targeted_refinement_runs"
    __table_args__ = (
        UniqueConstraint(
            "predictive_edge_run_id", name="uq_targeted_refinement_edge_run"
        ),
        Index("ix_targeted_refinement_experiment", "experiment_key", unique=True),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    predictive_edge_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    contract_version: Mapped[int] = mapped_column(Integer, nullable=False)
    conclusion: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TargetedRefinementCandidateRecord(Base):
    __tablename__ = "targeted_refinement_candidates"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "candidate_fingerprint", name="uq_targeted_refinement_candidate"
        ),
        Index("ix_targeted_refinement_run_status", "run_id", "status"),
        Index("ix_targeted_refinement_run_distance", "run_id", "normalized_distance"),
        Index("ix_targeted_refinement_seed", "seed_fingerprint"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("targeted_refinement_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    seed_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    normalized_distance: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    configuration: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    training_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    validation_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    test_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    passed_requirements: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    failed_requirements: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    requirement_margins: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )
    rejection_reason: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


def _json(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "total_seconds"):
        return int(value.total_seconds())
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    return value


def _metrics(item: TargetedCandidateResult, name: str) -> dict[str, object] | None:
    value = getattr(item, name)
    return cast(dict[str, object], _json(asdict(value))) if value is not None else None


async def persist_targeted_refinement_result(
    *,
    session: AsyncSession,
    predictive_edge_run_id: UUID,
    experiment_key: str,
    result: TargetedRefinementResult,
) -> UUID:
    """Atomically replace an incomplete stage graph with one complete result."""

    existing = await session.scalar(
        select(TargetedRefinementRunRecord).where(
            TargetedRefinementRunRecord.predictive_edge_run_id == predictive_edge_run_id
        )
    )
    if existing is not None:
        await session.execute(
            delete(TargetedRefinementRunRecord).where(
                TargetedRefinementRunRecord.id == existing.id
            )
        )
        await session.flush()
    run_id = uuid4()
    now = datetime.now(UTC)
    session.add(
        TargetedRefinementRunRecord(
            id=run_id,
            predictive_edge_run_id=predictive_edge_run_id,
            experiment_key=experiment_key,
            contract_version=TARGETED_REFINEMENT_CONTRACT_VERSION,
            conclusion=result.conclusion.value,
            summary=_json(asdict(result.summary)),
            created_at=now,
        )
    )
    await session.flush()
    for item in result.candidates:
        session.add(
            TargetedRefinementCandidateRecord(
                id=uuid4(),
                run_id=run_id,
                candidate_fingerprint=item.candidate.candidate_fingerprint,
                seed_fingerprint=item.configuration.seed_fingerprint,
                status=item.status.value,
                normalized_distance=item.normalized_distance_to_pass,
                configuration=_json(asdict(item.candidate)),
                training_metrics=_metrics(item, "training"),
                validation_metrics=_metrics(item, "validation"),
                test_metrics=_metrics(item, "test"),
                passed_requirements=list(item.passed_requirements),
                failed_requirements=list(item.failed_requirements),
                requirement_margins=_json(
                    [asdict(value) for value in item.requirement_margins]
                ),
                rejection_reason=item.rejection_reason,
                created_at=now,
            )
        )
    await session.commit()
    return run_id
