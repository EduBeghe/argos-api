from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
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
from app.market.catalog import Instrument
from app.optimization.composite_search import (
    COMPOSITE_FEATURE_POOL,
    COMPOSITE_FEATURE_POOL_VERSION,
    COMPOSITE_NORMALIZATION_VERSION,
    COMPOSITE_RANKING_VERSION,
    COMPOSITE_SEARCH_ALGORITHM_VERSION,
    CompositeSearchBudget,
    CompositeSearchResult,
)


class CompositeSearchRunRecord(Base):
    __tablename__ = "composite_search_runs"
    __table_args__ = (
        UniqueConstraint("predictive_edge_run_id", name="uq_composite_search_edge_run"),
        Index("ix_composite_search_experiment", "experiment_key", unique=True),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    predictive_edge_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    range_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    range_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    algorithm_version: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_pool_version: Mapped[int] = mapped_column(Integer, nullable=False)
    ranking_version: Mapped[int] = mapped_column(Integer, nullable=False)
    normalization_version: Mapped[int] = mapped_column(Integer, nullable=False)
    search_configuration: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False
    )
    feature_pool: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    funnel_by_depth: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )
    excluded_features: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False)
    conclusion: Mapped[str] = mapped_column(String(64), nullable=False)
    maximum_depth_reached: Mapped[int] = mapped_column(Integer, nullable=False)
    total_evaluations: Mapped[int] = mapped_column(Integer, nullable=False)
    computed: Mapped[int] = mapped_column(Integer, nullable=False)
    reused: Mapped[int] = mapped_column(Integer, nullable=False)
    budget_exhausted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    cheap_evaluations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    cheap_evaluations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    training_simulations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    training_simulations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_simulations_computed: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    validation_simulations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    test_simulations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    test_simulations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    best_depth: Mapped[int | None] = mapped_column(Integer)
    best_candidate_fingerprint: Mapped[str | None] = mapped_column(String(64))
    dominant_rejection_reason: Mapped[str | None] = mapped_column(String(128))
    best_economic_candidate_fingerprint: Mapped[str | None] = mapped_column(String(64))
    best_exploratory_candidate_fingerprint: Mapped[str | None] = mapped_column(
        String(64)
    )
    candidate_simulations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_simulations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    production_simulations_computed: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    production_simulations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    buy_and_hold_simulations_computed: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    buy_and_hold_simulations_reused: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    normalizations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    normalizations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    simulations_avoided: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class CompositeSearchCandidateRecord(Base):
    __tablename__ = "composite_search_candidates"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "candidate_fingerprint", name="uq_composite_search_candidate"
        ),
        Index("ix_composite_search_candidate_filter", "run_id", "depth", "status"),
        CheckConstraint(
            "status IN ('retained_training', 'rejected_validation', "
            "'passed_validation', 'rejected_test', 'confirmed_candidate')",
            name="ck_composite_search_candidate_status",
        ),
        CheckConstraint(
            "selection_track IS NULL OR selection_track IN ('economic', 'exploratory')",
            name="ck_composite_search_candidate_selection_track",
        ),
        CheckConstraint(
            "selection_track IS NULL OR selection_track = 'economic' "
            "OR status = 'retained_training'",
            name="ck_composite_search_exploratory_not_promoted",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("composite_search_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_fingerprint: Mapped[str | None] = mapped_column(String(64))
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    features: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    signed_weights: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )
    normalization_parameters: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False
    )
    configuration: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    ranking_breakdown: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    training_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    validation_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    test_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    passed_requirements: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    failed_requirements: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(String(128))
    normalized_distance: Mapped[str] = mapped_column(String(64), nullable=False)
    selection_track: Mapped[str | None] = mapped_column(String(16))
    exploration_reason: Mapped[str | None] = mapped_column(String(128))
    distance_to_training_qualification: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


def _json(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "total_seconds"):
        return int(value.total_seconds())
    return value


async def find_reusable_composite_search(
    session: AsyncSession, experiment_key: str
) -> CompositeSearchRunRecord | None:
    return cast(
        CompositeSearchRunRecord | None,
        await session.scalar(
            select(CompositeSearchRunRecord).where(
                CompositeSearchRunRecord.experiment_key == experiment_key,
                CompositeSearchRunRecord.status == "completed",
            )
        ),
    )


async def persist_composite_search_result(
    *,
    session: AsyncSession,
    predictive_edge_run_id: UUID,
    experiment_key: str,
    instrument: Instrument,
    dataset_fingerprint: str,
    range_start: datetime,
    range_end: datetime,
    budget: CompositeSearchBudget,
    result: CompositeSearchResult,
    started_at: datetime,
) -> UUID:
    existing = await session.scalar(
        select(CompositeSearchRunRecord).where(
            CompositeSearchRunRecord.predictive_edge_run_id == predictive_edge_run_id
        )
    )
    if existing is not None:
        await session.execute(
            delete(CompositeSearchRunRecord).where(
                CompositeSearchRunRecord.id == existing.id
            )
        )
        await session.flush()
    run_id = uuid4()
    now = datetime.now(UTC)
    session.add(
        CompositeSearchRunRecord(
            id=run_id,
            predictive_edge_run_id=predictive_edge_run_id,
            experiment_key=experiment_key,
            status="completed",
            asset=instrument.asset.value,
            provider=instrument.provider.value,
            market_type=instrument.market_type.value,
            interval=instrument.interval.value,
            dataset_fingerprint=dataset_fingerprint,
            range_start=range_start,
            range_end=range_end,
            algorithm_version=COMPOSITE_SEARCH_ALGORITHM_VERSION,
            feature_pool_version=COMPOSITE_FEATURE_POOL_VERSION,
            ranking_version=COMPOSITE_RANKING_VERSION,
            normalization_version=COMPOSITE_NORMALIZATION_VERSION,
            search_configuration=_json(asdict(budget)),
            feature_pool=list(COMPOSITE_FEATURE_POOL),
            funnel_by_depth=_json([asdict(item) for item in result.funnel_by_depth]),
            excluded_features=result.excluded_features,
            conclusion=result.conclusion.value,
            maximum_depth_reached=result.maximum_depth_reached,
            total_evaluations=result.total_evaluations,
            computed=result.computed,
            reused=result.reused,
            budget_exhausted=result.budget_exhausted,
            cheap_evaluations_computed=result.cheap_evaluations_computed,
            cheap_evaluations_reused=result.cheap_evaluations_reused,
            training_simulations_computed=result.training_simulations_computed,
            training_simulations_reused=result.training_simulations_reused,
            validation_simulations_computed=result.validation_simulations_computed,
            validation_simulations_reused=result.validation_simulations_reused,
            test_simulations_computed=result.test_simulations_computed,
            test_simulations_reused=result.test_simulations_reused,
            best_depth=result.best_depth,
            best_candidate_fingerprint=result.best_candidate_fingerprint,
            dominant_rejection_reason=result.dominant_rejection_reason,
            best_economic_candidate_fingerprint=(
                result.best_economic_candidate_fingerprint
            ),
            best_exploratory_candidate_fingerprint=(
                result.best_exploratory_candidate_fingerprint
            ),
            candidate_simulations_computed=result.candidate_simulations_computed,
            candidate_simulations_reused=result.candidate_simulations_reused,
            production_simulations_computed=result.production_simulations_computed,
            production_simulations_reused=result.production_simulations_reused,
            buy_and_hold_simulations_computed=(
                result.buy_and_hold_simulations_computed
            ),
            buy_and_hold_simulations_reused=result.buy_and_hold_simulations_reused,
            normalizations_computed=result.normalizations_computed,
            normalizations_reused=result.normalizations_reused,
            simulations_avoided=result.simulations_avoided,
            started_at=started_at,
            completed_at=now,
        )
    )
    await session.flush()
    for item in result.candidates:
        candidate = item.candidate
        session.add(
            CompositeSearchCandidateRecord(
                id=uuid4(),
                run_id=run_id,
                candidate_fingerprint=candidate.fingerprint,
                parent_fingerprint=item.parent_fingerprint,
                depth=item.depth,
                features=list(candidate.features),
                signed_weights=_json([asdict(value) for value in candidate.weights]),
                normalization_parameters=_json(
                    [asdict(value) for value in candidate.normalization]
                ),
                configuration=_json(asdict(candidate)),
                status=item.status.value,
                ranking_breakdown=_json(asdict(item.ranking)),
                training_metrics=_json(asdict(item.training_metrics)),
                validation_metrics=_json(asdict(item.validation_metrics))
                if item.validation_metrics
                else None,
                test_metrics=_json(asdict(item.test_metrics))
                if item.test_metrics
                else None,
                passed_requirements=list(item.passed_requirements),
                failed_requirements=list(item.failed_requirements),
                rejection_reason=item.rejection_reason,
                normalized_distance=str(item.normalized_distance),
                selection_track=item.selection_track.value,
                exploration_reason=item.exploration_reason,
                distance_to_training_qualification=str(
                    item.distance_to_training_qualification
                ),
                created_at=now,
            )
        )
    await session.commit()
    return run_id
