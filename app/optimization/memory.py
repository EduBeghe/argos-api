from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class StrategyRefinementRunRecord(Base):
    __tablename__ = "strategy_refinement_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed')",
            name="ck_strategy_refinement_runs_status",
        ),
        CheckConstraint(
            "(status = 'running' AND completed_at IS NULL AND failed_at IS NULL) OR "
            "(status = 'completed' AND completed_at IS NOT NULL "
            "AND failed_at IS NULL) OR "
            "(status = 'failed' AND failed_at IS NOT NULL AND completed_at IS NULL)",
            name="ck_strategy_refinement_runs_lifecycle",
        ),
        UniqueConstraint("experiment_key", name="uq_strategy_refinement_runs_key"),
        Index(
            "ix_strategy_refinement_runs_compatibility",
            "status",
            "score_model_name",
            "score_model_version",
            "strategy_name",
            "strategy_version",
            "simulation_engine_version",
            "search_planner_version",
            "started_at",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    requested_to: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    score_model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    score_model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    simulation_engine_version: Mapped[int] = mapped_column(Integer, nullable=False)
    initial_cash: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    account_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    slippage_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    decision_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    training_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    training_to: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    validation_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    validation_to: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    test_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_to: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    refinement_budget: Mapped[dict[str, object]] = mapped_column(
        JSON_TYPE, nullable=False
    )
    planner_allocation: Mapped[dict[str, object]] = mapped_column(
        JSON_TYPE, nullable=False
    )
    search_planner_version: Mapped[int] = mapped_column(Integer, nullable=False)
    history_cutoff: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    recommendation_status: Mapped[str | None] = mapped_column(String(64))
    recommended_candidate_id: Mapped[str | None] = mapped_column(String(64))
    recommendation_explanation: Mapped[str | None] = mapped_column(Text)
    counts: Mapped[dict[str, object] | None] = mapped_column(JSON_TYPE)
    search_plan: Mapped[dict[str, object] | None] = mapped_column(JSON_TYPE)
    result_payload: Mapped[dict[str, object] | None] = mapped_column(JSON_TYPE)
    elapsed_seconds: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(String(512))
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StrategyCandidateRecord(Base):
    __tablename__ = "strategy_candidates"
    stable_candidate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    score_model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    score_model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    score_parameters: Mapped[dict[str, object]] = mapped_column(
        JSON_TYPE, nullable=False
    )
    strategy_parameters: Mapped[dict[str, object]] = mapped_column(
        JSON_TYPE, nullable=False
    )
    canonical_parameters: Mapped[dict[str, object]] = mapped_column(
        JSON_TYPE, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class StrategyCandidateEvaluationRecord(Base):
    __tablename__ = "strategy_candidate_evaluations"
    __table_args__ = (
        CheckConstraint(
            "phase IN ('score_screening', 'joint_training', 'validation', 'test', "
            "'controlled_ablation')",
            name="ck_strategy_candidate_evaluations_phase",
        ),
        CheckConstraint(
            "segment IN ('screening', 'training', 'validation', 'test')",
            name="ck_strategy_candidate_evaluations_segment",
        ),
        UniqueConstraint(
            "refinement_run_id",
            "stable_candidate_id",
            "phase",
            "segment",
            name="uq_strategy_candidate_evaluations_identity",
        ),
        Index(
            "ix_strategy_candidate_evaluations_profile",
            "stable_candidate_id",
            "segment",
            "evaluated_at",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    refinement_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("strategy_refinement_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stable_candidate_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("strategy_candidates.stable_candidate_id"),
        nullable=False,
    )
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    segment: Mapped[str] = mapped_column(String(16), nullable=False)
    disabled_signal: Mapped[str | None] = mapped_column(String(32))
    total_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    absolute_pnl: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fees_paid: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    final_equity: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    return_over_drawdown: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    rank: Mapped[int | None] = mapped_column(Integer)
    eligible: Mapped[bool | None] = mapped_column(Boolean)
    rejection_reason: Mapped[str | None] = mapped_column(String(128))
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class StrategyBenchmarkEvaluationRecord(Base):
    __tablename__ = "strategy_benchmark_evaluations"
    __table_args__ = (
        CheckConstraint(
            "benchmark IN ('production_baseline', 'buy_and_hold')",
            name="ck_strategy_benchmark_evaluations_name",
        ),
        UniqueConstraint(
            "refinement_run_id",
            "benchmark",
            "segment",
            name="uq_strategy_benchmark_evaluations_identity",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    refinement_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("strategy_refinement_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    benchmark: Mapped[str] = mapped_column(String(32), nullable=False)
    segment: Mapped[str] = mapped_column(String(16), nullable=False)
    total_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    absolute_pnl: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fees_paid: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    final_equity: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    return_over_drawdown: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


async def find_experiment(
    session: AsyncSession, experiment_key: str
) -> StrategyRefinementRunRecord | None:
    return cast(
        StrategyRefinementRunRecord | None,
        await session.scalar(
            select(StrategyRefinementRunRecord).where(
                StrategyRefinementRunRecord.experiment_key == experiment_key
            )
        ),
    )


async def create_run(
    session: AsyncSession, values: dict[str, object]
) -> StrategyRefinementRunRecord | None:
    statement = (
        insert(StrategyRefinementRunRecord)
        .values(id=uuid4(), status="running", **values)
        .on_conflict_do_nothing(index_elements=["experiment_key"])
        .returning(StrategyRefinementRunRecord)
    )
    return (await session.scalars(statement)).one_or_none()


async def upsert_candidates(
    session: AsyncSession, values: Sequence[dict[str, object]]
) -> None:
    if values:
        await session.execute(
            insert(StrategyCandidateRecord)
            .values(list(values))
            .on_conflict_do_nothing(index_elements=["stable_candidate_id"])
        )


async def insert_candidate_evaluations(
    session: AsyncSession, values: Sequence[dict[str, object]]
) -> None:
    if values:
        await session.execute(
            insert(StrategyCandidateEvaluationRecord).values(list(values))
        )


async def insert_benchmark_evaluations(
    session: AsyncSession, values: Sequence[dict[str, object]]
) -> None:
    if values:
        await session.execute(
            insert(StrategyBenchmarkEvaluationRecord).values(list(values))
        )


async def complete_run(
    session: AsyncSession, run_id: UUID, values: dict[str, object]
) -> None:
    await session.execute(
        update(StrategyRefinementRunRecord)
        .where(
            StrategyRefinementRunRecord.id == run_id,
            StrategyRefinementRunRecord.status == "running",
        )
        .values(status="completed", **values)
    )


async def fail_run(
    session: AsyncSession, run_id: UUID, *, failed_at: datetime, reason: str
) -> None:
    await session.execute(
        update(StrategyRefinementRunRecord)
        .where(StrategyRefinementRunRecord.id == run_id)
        .values(status="failed", failed_at=failed_at, failure_reason=reason[:512])
    )


async def load_compatible_evidence(
    session: AsyncSession,
    *,
    score_model_name: str,
    score_model_version: int,
    strategy_name: str,
    strategy_version: int,
    simulation_engine_version: int,
    history_cutoff: datetime,
) -> list[
    tuple[StrategyCandidateEvaluationRecord, StrategyRefinementRunRecord, Decimal]
]:
    baseline = StrategyBenchmarkEvaluationRecord.__table__.alias("baseline")
    statement = (
        select(
            StrategyCandidateEvaluationRecord,
            StrategyRefinementRunRecord,
            baseline.c.total_return,
        )
        .join(
            StrategyRefinementRunRecord,
            StrategyRefinementRunRecord.id
            == StrategyCandidateEvaluationRecord.refinement_run_id,
        )
        .join(
            baseline,
            (baseline.c.refinement_run_id == StrategyRefinementRunRecord.id)
            & (baseline.c.benchmark == "production_baseline")
            & (baseline.c.segment == StrategyCandidateEvaluationRecord.segment),
        )
        .where(
            StrategyRefinementRunRecord.status == "completed",
            StrategyRefinementRunRecord.score_model_name == score_model_name,
            StrategyRefinementRunRecord.score_model_version == score_model_version,
            StrategyRefinementRunRecord.strategy_name == strategy_name,
            StrategyRefinementRunRecord.strategy_version == strategy_version,
            StrategyRefinementRunRecord.simulation_engine_version
            == simulation_engine_version,
            StrategyRefinementRunRecord.completed_at <= history_cutoff,
            StrategyCandidateEvaluationRecord.phase.in_(
                ("joint_training", "validation", "test")
            ),
        )
    )
    return [
        (evaluation, run, baseline_return)
        for evaluation, run, baseline_return in (await session.execute(statement)).all()
    ]


async def load_candidate_definitions(
    session: AsyncSession, candidate_ids: Sequence[str]
) -> dict[str, StrategyCandidateRecord]:
    if not candidate_ids:
        return {}
    records = (
        await session.scalars(
            select(StrategyCandidateRecord).where(
                StrategyCandidateRecord.stable_candidate_id.in_(candidate_ids)
            )
        )
    ).all()
    return {record.stable_candidate_id: record for record in records}


async def refinement_storage_diagnostics(
    session: AsyncSession,
) -> dict[str, int]:
    counts = {}
    for name, model in (
        ("refinement_runs", StrategyRefinementRunRecord),
        ("candidates", StrategyCandidateRecord),
        ("evaluations", StrategyCandidateEvaluationRecord),
        ("benchmarks", StrategyBenchmarkEvaluationRecord),
    ):
        counts[name] = int(
            await session.scalar(select(func.count()).select_from(model)) or 0
        )
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        counts["table_and_index_bytes"] = int(
            await session.scalar(
                text(
                    "SELECT sum(pg_total_relation_size(name::regclass)) "
                    "FROM unnest(ARRAY['strategy_refinement_runs', "
                    "'strategy_candidates', 'strategy_candidate_evaluations', "
                    "'strategy_benchmark_evaluations']) AS name"
                )
            )
            or 0
        )
    return counts
