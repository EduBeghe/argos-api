
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy import (
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
    select,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import HistoricalDataset
from app.optimization.memory import (
    StrategyRefinementRunRecord,
    upsert_candidates,
)
from app.optimization.refinement import (
    RefinementBudget,
    RefinementCandidate,
    canonical_candidate_parameters,
)
from app.optimization.simulation import DEFAULT_DECISION_INTERVAL
from app.optimization.walk_forward import (
    WALK_FORWARD_ALGORITHM_VERSION,
    WalkForwardBenchmarkEvaluation,
    WalkForwardCandidateEvaluation,
    WalkForwardCandidateSelection,
    WalkForwardClassification,
    WalkForwardConfiguration,
    WalkForwardResult,
    completed_window_count,
    evaluate_walk_forward,
)
from app.scores.calculation import MODEL_NAME, MODEL_VERSION
from app.strategies.baseline_long_only import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
)


class WalkForwardMemoryError(RuntimeError):
    """Raised when persisted walk-forward memory cannot be used safely."""


class WalkForwardRunRecord(Base):
    __tablename__ = "walk_forward_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('completed', 'insufficient_history')",
            name="ck_walk_forward_runs_status",
        ),
        UniqueConstraint("experiment_key", name="uq_walk_forward_runs_experiment_key"),
        Index(
            "ix_walk_forward_runs_refinement_started",
            "refinement_run_id",
            "started_at",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    refinement_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("strategy_refinement_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    algorithm_version: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    training_days: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_days: Mapped[int] = mapped_column(Integer, nullable=False)
    test_days: Mapped[int] = mapped_column(Integer, nullable=False)
    step_days: Mapped[int] = mapped_column(Integer, nullable=False)
    maximum_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    maximum_candidates: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_valid_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_profitable_window_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    maximum_drawdown: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    maximum_return_standard_deviation: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    initial_cash: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    slippage_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    decision_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    windows_generated: Mapped[int] = mapped_column(Integer, nullable=False)
    windows_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    candidates_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_computed: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluations_reused: Mapped[int] = mapped_column(Integer, nullable=False)
    compute_avoided_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    recommended_candidate_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("strategy_candidates.stable_candidate_id"),
    )
    recommendation_explanation: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    elapsed_seconds: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)


class WalkForwardWindowRecord(Base):
    __tablename__ = "walk_forward_windows"
    __table_args__ = (
        UniqueConstraint(
            "walk_forward_run_id", "ordinal", name="uq_walk_forward_windows_ordinal"
        ),
        CheckConstraint("ordinal > 0", name="ck_walk_forward_windows_ordinal"),
        CheckConstraint(
            "training_from < training_to AND training_to = validation_from "
            "AND validation_from < validation_to AND validation_to = test_from "
            "AND test_from < test_to",
            name="ck_walk_forward_windows_chronology",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    walk_forward_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
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
    decision_count: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    selection_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)


class WalkForwardCandidateSelectionRecord(Base):
    __tablename__ = "walk_forward_candidate_selections"
    __table_args__ = (
        CheckConstraint(
            "validation_rank > 0",
            name="ck_walk_forward_candidate_selections_rank",
        ),
        CheckConstraint(
            "selection_reason IN ('selected_by_window_validation_rank', "
            "'outside_window_test_candidate_limit')",
            name="ck_walk_forward_candidate_selections_reason",
        ),
        CheckConstraint(
            "(selected_for_test AND selection_reason = "
            "'selected_by_window_validation_rank') OR "
            "(NOT selected_for_test AND selection_reason = "
            "'outside_window_test_candidate_limit')",
            name="ck_walk_forward_candidate_selections_outcome",
        ),
        UniqueConstraint(
            "walk_forward_run_id",
            "window_id",
            "stable_candidate_id",
            name="uq_walk_forward_candidate_selections_identity",
        ),
        Index(
            "ix_walk_forward_candidate_selections_page",
            "walk_forward_run_id",
            "stable_candidate_id",
            "window_id",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    walk_forward_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_windows.id", ondelete="CASCADE"),
        nullable=False,
    )
    stable_candidate_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("strategy_candidates.stable_candidate_id"),
        nullable=False,
    )
    validation_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_for_test: Mapped[bool] = mapped_column(Boolean, nullable=False)
    selection_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    validation_total_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    validation_absolute_pnl: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    validation_max_drawdown: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    validation_final_equity: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    validation_fees_paid: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    validation_trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_return_over_drawdown: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )


class WalkForwardCandidateEvaluationRecord(Base):
    __tablename__ = "walk_forward_candidate_evaluations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('completed', 'failed')",
            name="ck_walk_forward_candidate_evaluations_status",
        ),
        CheckConstraint(
            "(status = 'completed' AND total_return IS NOT NULL "
            "AND absolute_pnl IS NOT NULL AND max_drawdown IS NOT NULL "
            "AND final_equity IS NOT NULL AND fees_paid IS NOT NULL "
            "AND trade_count IS NOT NULL AND return_over_drawdown IS NOT NULL "
            "AND failure_reason IS NULL) OR "
            "(status = 'failed' AND total_return IS NULL "
            "AND absolute_pnl IS NULL AND max_drawdown IS NULL "
            "AND final_equity IS NULL AND fees_paid IS NULL "
            "AND trade_count IS NULL AND return_over_drawdown IS NULL "
            "AND failure_reason IS NOT NULL)",
            name="ck_walk_forward_candidate_evaluations_lifecycle",
        ),
        UniqueConstraint(
            "walk_forward_run_id",
            "window_id",
            "stable_candidate_id",
            name="uq_walk_forward_candidate_evaluations_identity",
        ),
        Index(
            "ix_walk_forward_candidate_evaluations_page",
            "walk_forward_run_id",
            "stable_candidate_id",
            "window_id",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    walk_forward_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_windows.id", ondelete="CASCADE"),
        nullable=False,
    )
    stable_candidate_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("strategy_candidates.stable_candidate_id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    total_return: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    absolute_pnl: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    final_equity: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    fees_paid: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    trade_count: Mapped[int | None] = mapped_column(Integer)
    return_over_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    failure_reason: Mapped[str | None] = mapped_column(String(128))


class WalkForwardBenchmarkEvaluationRecord(Base):
    __tablename__ = "walk_forward_benchmark_evaluations"
    __table_args__ = (
        CheckConstraint(
            "benchmark IN ('production_baseline', 'buy_and_hold')",

            name="ck_walk_forward_benchmark_evaluations_name",
        ),
        CheckConstraint(
            "status IN ('completed', 'failed')",
            name="ck_walk_forward_benchmark_evaluations_status",
        ),
        CheckConstraint(
            "(status = 'completed' AND total_return IS NOT NULL "
            "AND absolute_pnl IS NOT NULL AND max_drawdown IS NOT NULL "
            "AND final_equity IS NOT NULL AND fees_paid IS NOT NULL "
            "AND trade_count IS NOT NULL AND return_over_drawdown IS NOT NULL "
            "AND failure_reason IS NULL) OR "
            "(status = 'failed' AND total_return IS NULL "
            "AND absolute_pnl IS NULL AND max_drawdown IS NULL "
            "AND final_equity IS NULL AND fees_paid IS NULL "
            "AND trade_count IS NULL AND return_over_drawdown IS NULL "
            "AND failure_reason IS NOT NULL)",
            name="ck_walk_forward_benchmark_evaluations_lifecycle",
        ),
        UniqueConstraint(
            "walk_forward_run_id",
            "window_id",
            "benchmark",
            name="uq_walk_forward_benchmark_evaluations_identity",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    walk_forward_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_windows.id", ondelete="CASCADE"),
        nullable=False,
    )
    benchmark: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    total_return: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    absolute_pnl: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    final_equity: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    fees_paid: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    trade_count: Mapped[int | None] = mapped_column(Integer)
    return_over_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    failure_reason: Mapped[str | None] = mapped_column(String(128))


class WalkForwardCandidateStabilityRecord(Base):
    __tablename__ = "walk_forward_candidate_stability"
    __table_args__ = (
        CheckConstraint(
            "classification IN ('stable_candidate', 'unstable_candidate', "
            "'consistently_unprofitable', 'insufficient_evidence', "
            "'evaluation_failed')",
            name="ck_walk_forward_candidate_stability_classification",
        ),
        UniqueConstraint(
            "walk_forward_run_id",
            "stable_candidate_id",
            name="uq_walk_forward_candidate_stability_identity",
        ),
        Index(
            "ix_walk_forward_candidate_stability_classification",
            "walk_forward_run_id",
            "classification",
            "stable_candidate_id",
        ),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    walk_forward_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("walk_forward_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stable_candidate_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("strategy_candidates.stable_candidate_id"),
        nullable=False,
    )
    ranking: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    windows_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    profitable_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    losing_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    profitable_window_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    median_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    worst_window_return: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    best_window_return: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    return_standard_deviation: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    mean_max_drawdown: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    worst_max_drawdown: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    aggregate_fees_paid: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    aggregate_trade_count: Mapped[int] = mapped_column(Integer, nullable=False)
    positive_excess_return_vs_production_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    positive_excess_return_vs_buy_and_hold_ratio: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    explanation: Mapped[str] = mapped_column(Text, nullable=False)


@dataclass(frozen=True, slots=True)
class WalkForwardExecution:
    run_id: UUID
    result: WalkForwardResult | None
    reused_existing_experiment: bool
    evaluations_computed: int
    evaluations_reused: int
    compute_avoided_estimate: int


async def execute_walk_forward(
    *,
    session: AsyncSession,
    dataset: HistoricalDataset,
    refinement_run_id: UUID,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    configuration: WalkForwardConfiguration,
    started_at: datetime,
    decision_interval: timedelta = DEFAULT_DECISION_INTERVAL,
) -> WalkForwardExecution:
    """Evaluate and atomically persist one reusable walk-forward experiment."""
    if started_at.tzinfo is not UTC:
        raise WalkForwardMemoryError("walk-forward started_at must use UTC")
    source = await session.get(StrategyRefinementRunRecord, refinement_run_id)
    if source is None or source.status != "completed":
        raise WalkForwardMemoryError(
            "walk-forward requires a completed persisted refinement run"
        )
    if source.dataset_fingerprint != dataset.fingerprint:
        raise WalkForwardMemoryError(
            "walk-forward dataset does not match the source refinement run"
        )
    key = _experiment_key(
        dataset,
        refinement_run_id,

        initial_cash,
        fee_rate,
        slippage_rate,
        decision_interval,
        configuration,
    )
    existing = await session.scalar(
        select(WalkForwardRunRecord).where(WalkForwardRunRecord.experiment_key == key)
    )
    if existing is not None:
        avoided = existing.evaluations_computed
        return WalkForwardExecution(existing.id, None, True, 0, avoided, avoided)
    refinement_budget = _refinement_budget(source.refinement_budget)
    started = perf_counter()
    result = evaluate_walk_forward(
        dataset=dataset,
        feature_matrix=build_historical_feature_matrix(dataset),
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        refinement_budget=refinement_budget,
        configuration=configuration,
        decision_interval=decision_interval,
    )
    run_id = uuid4()
    completed_at = datetime.now(UTC)
    run = WalkForwardRunRecord(
        id=run_id,
        experiment_key=key,
        refinement_run_id=refinement_run_id,
        status=result.conclusion.value,
        algorithm_version=WALK_FORWARD_ALGORITHM_VERSION,
        dataset_fingerprint=dataset.fingerprint,
        **asdict(configuration),
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        decision_interval_seconds=int(decision_interval.total_seconds()),
        windows_generated=len(result.windows),
        windows_completed=completed_window_count(result),
        candidates_evaluated=len(result.stability),
        evaluations_computed=result.evaluations_computed,
        evaluations_reused=0,
        compute_avoided_estimate=0,
        recommended_candidate_id=result.recommended_candidate_id,
        recommendation_explanation=result.recommendation_explanation,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=Decimal(str(perf_counter() - started)),
    )
    session.add(run)
    window_ids = {window.ordinal: uuid4() for window in result.windows}
    session.add_all(
        WalkForwardWindowRecord(
            id=window_ids[window.ordinal],
            walk_forward_run_id=run_id,
            **asdict(window),
        )
        for window in result.windows
    )
    candidates = tuple(
        {
            selection.candidate.stable_candidate_id: selection.candidate
            for selection in result.selections
        }.values()
    )
    await upsert_candidates(
        session,
        tuple(_candidate_definition(candidate, started_at) for candidate in candidates),
    )
    session.add_all(
        _selection_record(run_id, window_ids, selection)
        for selection in result.selections
    )
    session.add_all(
        _candidate_evaluation_record(run_id, window_ids, evaluation)
        for evaluation in result.candidate_evaluations
    )
    session.add_all(
        _benchmark_record(run_id, window_ids, benchmark)
        for benchmark in result.benchmark_evaluations
    )
    ranked = sorted(
        result.stability,
        key=lambda item: (
            item.classification is not WalkForwardClassification.STABLE_CANDIDATE,
            -item.median_return,
            item.worst_max_drawdown,
            item.candidate.stable_candidate_id,
        ),
    )
    session.add_all(
        WalkForwardCandidateStabilityRecord(
            id=uuid4(),
            walk_forward_run_id=run_id,
            stable_candidate_id=value.candidate.stable_candidate_id,
            ranking=rank,
            **{
                name: getattr(value, name)
                for name in (
                    "classification",
                    "windows_evaluated",
                    "profitable_windows",
                    "losing_windows",
                    "profitable_window_ratio",
                    "mean_return",
                    "median_return",
                    "worst_window_return",
                    "best_window_return",
                    "return_standard_deviation",
                    "mean_max_drawdown",
                    "worst_max_drawdown",
                    "aggregate_fees_paid",
                    "aggregate_trade_count",
                    "positive_excess_return_vs_production_ratio",
                    "positive_excess_return_vs_buy_and_hold_ratio",
                    "explanation",
                )
            },
        )
        for rank, value in enumerate(ranked, start=1)
    )
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        concurrent = await session.scalar(
            select(WalkForwardRunRecord).where(
                WalkForwardRunRecord.experiment_key == key
            )
        )
        if concurrent is None:
            raise
        avoided = concurrent.evaluations_computed
        return WalkForwardExecution(concurrent.id, None, True, 0, avoided, avoided)
    return WalkForwardExecution(
        run_id,
        result,
        False,
        result.evaluations_computed,
        0,
        0,
    )


def _refinement_budget(value: dict[str, object]) -> RefinementBudget:
    try:
        return RefinementBudget(
            score_screening_top_k=int(str(value["score_screening_top_k"])),
            validation_top_k=int(str(value["validation_top_k"])),

            max_score_configurations=int(str(value["max_score_configurations"])),
            max_joint_candidates=int(str(value["max_joint_candidates"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WalkForwardMemoryError(
            "source refinement run has an invalid persisted budget"
        ) from error


def _experiment_key(
    dataset: HistoricalDataset,
    refinement_run_id: UUID,
    initial_cash: Decimal,
    fee_rate: Decimal,
    slippage_rate: Decimal,
    decision_interval: timedelta,
    configuration: WalkForwardConfiguration,
) -> str:
    payload = {
        "algorithm_version": WALK_FORWARD_ALGORITHM_VERSION,
        "dataset_fingerprint": dataset.fingerprint,
        "refinement_run_id": str(refinement_run_id),
        "initial_cash": str(initial_cash),
        "fee_rate": str(fee_rate),
        "slippage_rate": str(slippage_rate),
        "decision_interval_seconds": int(decision_interval.total_seconds()),
        "configuration": {
            name: str(value) if isinstance(value, Decimal) else value
            for name, value in asdict(configuration).items()
        },
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_evaluation_record(
    run_id: UUID,
    window_ids: dict[int, UUID],
    evaluation: WalkForwardCandidateEvaluation,
) -> WalkForwardCandidateEvaluationRecord:
    metrics = evaluation.metrics
    return WalkForwardCandidateEvaluationRecord(
        id=uuid4(),
        walk_forward_run_id=run_id,
        window_id=window_ids[evaluation.window.ordinal],
        stable_candidate_id=evaluation.candidate.stable_candidate_id,
        status=evaluation.status.value,
        total_return=metrics.total_return if metrics else None,
        absolute_pnl=metrics.absolute_pnl if metrics else None,
        max_drawdown=metrics.max_drawdown if metrics else None,
        final_equity=metrics.final_equity if metrics else None,
        fees_paid=metrics.fees_paid if metrics else None,
        trade_count=metrics.trade_count if metrics else None,
        return_over_drawdown=metrics.return_over_drawdown if metrics else None,
        failure_reason=evaluation.failure_reason,
    )


def _selection_record(
    run_id: UUID,
    window_ids: dict[int, UUID],
    selection: WalkForwardCandidateSelection,
) -> WalkForwardCandidateSelectionRecord:
    metrics = selection.validation_metrics
    return WalkForwardCandidateSelectionRecord(
        id=uuid4(),
        walk_forward_run_id=run_id,
        window_id=window_ids[selection.window.ordinal],
        stable_candidate_id=selection.candidate.stable_candidate_id,
        validation_rank=selection.validation_rank,
        selected_for_test=selection.selected_for_test,
        selection_reason=selection.selection_reason,
        validation_total_return=metrics.total_return,
        validation_absolute_pnl=metrics.absolute_pnl,
        validation_max_drawdown=metrics.max_drawdown,
        validation_final_equity=metrics.final_equity,
        validation_fees_paid=metrics.fees_paid,
        validation_trade_count=metrics.trade_count,
        validation_return_over_drawdown=metrics.return_over_drawdown,
    )


def _benchmark_record(
    run_id: UUID,
    window_ids: dict[int, UUID],
    evaluation: WalkForwardBenchmarkEvaluation,
) -> WalkForwardBenchmarkEvaluationRecord:
    metrics = evaluation.metrics
    return WalkForwardBenchmarkEvaluationRecord(
        id=uuid4(),
        walk_forward_run_id=run_id,
        window_id=window_ids[evaluation.window.ordinal],
        benchmark=evaluation.benchmark,
        status=evaluation.status.value,
        total_return=metrics.total_return if metrics else None,
        absolute_pnl=metrics.absolute_pnl if metrics else None,
        max_drawdown=metrics.max_drawdown if metrics else None,
        final_equity=metrics.final_equity if metrics else None,
        fees_paid=metrics.fees_paid if metrics else None,
        trade_count=metrics.trade_count if metrics else None,
        return_over_drawdown=metrics.return_over_drawdown if metrics else None,
        failure_reason=evaluation.failure_reason,
    )


def _candidate_definition(
    candidate: RefinementCandidate, created_at: datetime
) -> dict[str, object]:
    score = {
        name: str(getattr(candidate.score_parameters, name))
        for name in (
            "trend_weight",
            "rsi_weight",
            "return_1_weight",
            "return_12_weight",
            "volume_confirmation_weight",
        )
    }
    strategy = {
        name: str(getattr(candidate.strategy_parameters, name))
        for name in (
            "entry_threshold",
            "exit_threshold",
            "open_position_target_percentage",
        )
    }
    return {
        "stable_candidate_id": candidate.stable_candidate_id,
        "score_model_name": MODEL_NAME,
        "score_model_version": MODEL_VERSION,
        "strategy_name": STRATEGY_NAME,
        "strategy_version": STRATEGY_VERSION,
        "score_parameters": score,
        "strategy_parameters": strategy,
        "canonical_parameters": {
            "ordered_values": list(
                canonical_candidate_parameters(
                    candidate.score_parameters, candidate.strategy_parameters
                )
            )
        },
        "created_at": created_at,
    }
