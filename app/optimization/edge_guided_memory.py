import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.base import Base
from app.market.catalog import Instrument
from app.optimization.edge_guided import (
    EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
    CandidateFamily,
    EdgeGuidedCandidate,
    EdgeGuidedCandidateEvaluation,
    EdgeGuidedCandidateStatus,
    EdgeGuidedEvaluationResult,
    EdgeGuidedRanking,
    EdgeGuidedRejectionReason,
    EdgeGuidedSegmentEvaluation,
    EdgeGuidedStageEvidence,
    EdgeGuidedWindowEvaluation,
    EntryRule,
    ExitRule,
    ResearchOutcome,
)
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    SuggestedAction,
)
from app.optimization.refinement import RefinementSegmentName
from app.optimization.walk_forward import WalkForwardConfiguration, WalkForwardWindow
from app.paper.config import SimulationConfiguration

EDGE_GUIDED_PERSISTENCE_CONTRACT_VERSION = 2


class EdgeGuidedRunRecord(Base):
    __tablename__ = "edge_guided_evaluation_runs"
    __table_args__ = (
        UniqueConstraint("experiment_key", name="uq_edge_guided_runs_experiment_key"),
        Index("ix_edge_guided_runs_status", "status", "started_at"),
        Index("uq_edge_guided_runs_reuse_key", "reuse_key", unique=True),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    experiment_key: Mapped[str] = mapped_column(String(64), nullable=False)
    reuse_key: Mapped[str | None] = mapped_column(String(64))
    dataset_revision: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    predictive_edge_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("predictive_edge_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    range_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    range_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    initial_cash: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    account_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    slippage_rate: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    decision_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    maximum_candidates: Mapped[int] = mapped_column(Integer, nullable=False)
    algorithm_version: Mapped[int] = mapped_column(Integer, nullable=False)
    qualification_contract_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
    )
    generated_count: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluated_count: Mapped[int] = mapped_column(Integer, nullable=False)
    windows_available: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    research_outcome: Mapped[str | None] = mapped_column(String(64))
    outcome_explanation: Mapped[str | None] = mapped_column(Text)
    funnel: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    timings: Mapped[dict[str, object] | None] = mapped_column(JSONB)


class EdgeGuidedCandidateRecord(Base):
    __tablename__ = "edge_guided_candidate_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "candidate_fingerprint", name="uq_edge_guided_candidates_identity"
        ),
        Index("ix_edge_guided_candidates_status", "run_id", "status"),
        Index("ix_edge_guided_candidates_fingerprint", "candidate_fingerprint"),
        Index("ix_edge_guided_candidates_hypothesis", "source_hypothesis_id"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("edge_guided_evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False)
    predictor_name: Mapped[str] = mapped_column(String(64), nullable=False)
    predictor_version: Mapped[int] = mapped_column(Integer, nullable=False)
    condition: Mapped[str] = mapped_column(String(32), nullable=False)
    bin_lower_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    bin_upper_bound: Mapped[Decimal | None] = mapped_column(Numeric(38, 18))
    suggested_action: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    direction_mapping: Mapped[str] = mapped_column(String(32), nullable=False)
    score_model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    score_model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy_configuration_version: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    rejection_reasons: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    valid_window_count: Mapped[int] = mapped_column(Integer, nullable=False)
    confirmed_window_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rejected_validation_window_count: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    rejected_test_window_count: Mapped[int] = mapped_column(Integer, nullable=False)
    training_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    validation_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    test_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    unconditional_benchmark: Mapped[str] = mapped_column(String(32), nullable=False)


class EdgeGuidedWindowRecord(Base):
    __tablename__ = "edge_guided_window_evidence"
    __table_args__ = (
        UniqueConstraint(
            "candidate_evaluation_id",
            "window_ordinal",
            name="uq_edge_guided_windows_identity",
        ),
        Index("ix_edge_guided_windows_cache_key", "cache_key"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    candidate_evaluation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("edge_guided_candidate_evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_key: Mapped[str | None] = mapped_column(String(64))
    volatility_threshold: Mapped[Decimal] = mapped_column(
        Numeric(38, 18), nullable=False
    )
    training_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    validation_metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    test_metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    test_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejection_reasons: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )


class EdgeGuidedStageEvidenceRecord(Base):
    __tablename__ = "edge_guided_stage_evidence"
    __table_args__ = (
        UniqueConstraint("run_id", "cache_key", name="uq_edge_guided_stage_run_key"),
        Index("ix_edge_guided_stage_cache_key", "cache_key"),
        Index("ix_edge_guided_stage_candidate", "candidate_fingerprint", "stage"),
    )
    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("edge_guided_evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(16), nullable=False)
    window_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    segment_to: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    content_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


@dataclass(frozen=True, slots=True)
class EdgeGuidedExecution:
    run_id: UUID | None
    experiment_key: str
    reused_existing_experiment: bool
    result: EdgeGuidedEvaluationResult
    evaluations_reused: int
    evaluations_computed: int
    compute_avoided_estimate: int


def edge_guided_experiment_key(
    *,
    predictive_edge_run_id: UUID,
    dataset_fingerprint: str,
    instrument: Instrument,
    range_start: datetime,
    range_end: datetime,
    windows: WalkForwardConfiguration,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    decision_interval: timedelta,
    maximum_candidates: int,
    hypothesis_fingerprints: tuple[str, ...],
) -> str:
    payload = {
        "persistence_contract_version": EDGE_GUIDED_PERSISTENCE_CONTRACT_VERSION,
        "qualification_contract_version": EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
        "predictive_edge_run_id": str(predictive_edge_run_id),
        "dataset_fingerprint": dataset_fingerprint,
        "instrument": [
            instrument.asset.value,
            instrument.provider.value,
            instrument.market_type.value,
            instrument.interval.value,
        ],
        "range": [range_start.isoformat(), range_end.isoformat()],
        "windows": asdict(windows),
        "initial_cash": str(initial_cash),
        "account_currency": simulation.account_currency,
        "fee_rate": str(simulation.fee_rate),
        "slippage_rate": str(simulation.slippage_rate),
        "decision_interval_seconds": int(decision_interval.total_seconds()),
        "maximum_candidates": maximum_candidates,
        "hypotheses": sorted(hypothesis_fingerprints),
    }
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def find_reusable_edge_guided_run(
    session: AsyncSession, experiment_key: str
) -> EdgeGuidedExecution | None:
    run = await session.scalar(
        select(EdgeGuidedRunRecord).where(
            EdgeGuidedRunRecord.experiment_key == experiment_key,
            EdgeGuidedRunRecord.status == "completed",
            EdgeGuidedRunRecord.qualification_contract_version
            == EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
        )
    )
    if run is None:
        return None
    result = await _load_valid_result(session, run)
    if result is None:
        return None
    return _reused_execution(run, result)


async def find_reusable_weekly_research_run(
    session: AsyncSession, reuse_key: str
) -> EdgeGuidedExecution | None:
    """Short-circuit a complete research run before loading historical candles."""

    run = await session.scalar(
        select(EdgeGuidedRunRecord).where(
            EdgeGuidedRunRecord.reuse_key == reuse_key,
            EdgeGuidedRunRecord.status == "completed",
            EdgeGuidedRunRecord.qualification_contract_version
            == EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
        )
    )
    if run is None:
        return None
    result = await _load_valid_result(session, run)
    return _reused_execution(run, result) if result is not None else None


async def load_reusable_edge_guided_stages(
    session: AsyncSession,
    candidate_fingerprints: tuple[str, ...],
    windows: tuple[WalkForwardWindow, ...],
) -> dict[str, EdgeGuidedStageEvidence]:
    """Load complete stage evidence only from completed compatible runs."""

    if not candidate_fingerprints or not windows:
        return {}
    starts = tuple(
        value
        for window in windows
        for value in (window.training_from, window.validation_from, window.test_from)
    )
    ends = tuple(
        value
        for window in windows
        for value in (window.training_to, window.validation_to, window.test_to)
    )
    records = (
        await session.scalars(
            select(EdgeGuidedStageEvidenceRecord)
            .join(
                EdgeGuidedRunRecord,
                EdgeGuidedRunRecord.id == EdgeGuidedStageEvidenceRecord.run_id,
            )
            .where(
                EdgeGuidedStageEvidenceRecord.candidate_fingerprint.in_(
                    candidate_fingerprints
                ),
                EdgeGuidedStageEvidenceRecord.segment_from.in_(starts),
                EdgeGuidedStageEvidenceRecord.segment_to.in_(ends),
                EdgeGuidedRunRecord.status == "completed",
                EdgeGuidedRunRecord.qualification_contract_version
                == EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
            )
            .order_by(EdgeGuidedRunRecord.completed_at.desc())
        )
    ).all()
    reusable: dict[str, EdgeGuidedStageEvidence] = {}
    for item in records:
        if item.cache_key in reusable:
            continue
        reusable[item.cache_key] = EdgeGuidedStageEvidence(
            item.cache_key,
            item.candidate_fingerprint,
            RefinementSegmentName(item.stage),
            item.window_ordinal,
            item.segment_from,
            item.segment_to,
            item.content_fingerprint,
            _read_metric(item.metrics),
        )
    return reusable


def _reused_execution(
    run: EdgeGuidedRunRecord, result: EdgeGuidedEvaluationResult
) -> EdgeGuidedExecution:
    return EdgeGuidedExecution(
        run.id,
        run.experiment_key,
        True,
        result,
        result.evaluated_count,
        0,
        int(result.funnel.get("edges_evaluated", 0))
        + result.evaluated_count * run.windows_available,
    )


async def persist_edge_guided_result(
    *,
    session: AsyncSession,
    experiment_key: str,
    predictive_edge_run_id: UUID,
    instrument: Instrument,
    range_start: datetime,
    range_end: datetime,
    initial_cash: Decimal,
    simulation: SimulationConfiguration,
    decision_interval: timedelta,
    started_at: datetime,
    result: EdgeGuidedEvaluationResult,
    reuse_key: str | None = None,
    dataset_revision: dict[str, object] | None = None,
) -> EdgeGuidedExecution:
    existing = await session.scalar(
        select(EdgeGuidedRunRecord)
        .where(EdgeGuidedRunRecord.experiment_key == experiment_key)
        .with_for_update()
    )
    if existing is not None:
        valid_result = (
            await _load_valid_result(session, existing)
            if existing.status == "completed"
            else None
        )
        if valid_result is not None:
            return _reused_execution(existing, valid_result)
        await session.execute(
            delete(EdgeGuidedRunRecord).where(EdgeGuidedRunRecord.id == existing.id)
        )
        try:
            await session.flush()
        except Exception:
            await session.rollback()
            raise
    run_id = uuid4()
    session.add(
        EdgeGuidedRunRecord(
            id=run_id,
            experiment_key=experiment_key,
            reuse_key=reuse_key,
            dataset_revision=_json(dataset_revision)
            if dataset_revision is not None
            else None,
            predictive_edge_run_id=predictive_edge_run_id,
            status="completed",
            dataset_fingerprint=result.dataset_fingerprint,
            asset=instrument.asset.value,
            provider=instrument.provider.value,
            market_type=instrument.market_type.value,
            interval=instrument.interval.value,
            range_start=range_start,
            range_end=range_end,
            initial_cash=initial_cash,
            account_currency=simulation.account_currency,
            fee_rate=simulation.fee_rate,
            slippage_rate=simulation.slippage_rate,
            decision_interval_seconds=int(decision_interval.total_seconds()),
            maximum_candidates=result.maximum_candidates,
            algorithm_version=EDGE_GUIDED_PERSISTENCE_CONTRACT_VERSION,
            qualification_contract_version=result.qualification_contract_version,
            generated_count=result.generated_count,
            evaluated_count=result.evaluated_count,
            windows_available=result.windows_available,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            research_outcome=result.outcome.value,
            outcome_explanation=result.outcome_explanation,
            funnel=_json(result.funnel),
            timings=_json(result.timings),
        )
    )
    # Explicit flushes preserve FK insertion order without introducing ORM
    # relationships solely for persistence orchestration.
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        reused = await find_reusable_edge_guided_run(session, experiment_key)
        if reused is None:
            raise
        return reused
    except Exception:
        await session.rollback()
        raise
    for evidence in result.stage_evidence:
        session.add(
            EdgeGuidedStageEvidenceRecord(
                id=uuid4(),
                run_id=run_id,
                cache_key=evidence.cache_key,
                candidate_fingerprint=evidence.candidate_fingerprint,
                stage=evidence.stage.value,
                window_ordinal=evidence.window_ordinal,
                segment_from=evidence.segment_from,
                segment_to=evidence.segment_to,
                content_fingerprint=evidence.content_fingerprint,
                metrics=_metric(evidence.metrics),
            )
        )
    for evaluation in result.evaluations:
        candidate_id = uuid4()
        session.add(
            EdgeGuidedCandidateRecord(
                id=candidate_id,
                run_id=run_id,
                candidate_fingerprint=evaluation.candidate.candidate_fingerprint,
                source_hypothesis_id=evaluation.candidate.source_hypothesis_id,
                predictor_name=evaluation.candidate.predictor_name,
                predictor_version=evaluation.candidate.predictor_version,
                condition=evaluation.candidate.condition.value,
                bin_lower_bound=evaluation.candidate.lower_bound,
                bin_upper_bound=evaluation.candidate.upper_bound,
                suggested_action=evaluation.candidate.suggested_action.value,
                horizon_seconds=int(evaluation.candidate.horizon.total_seconds()),
                direction_mapping=evaluation.candidate.direction_mapping.value,
                score_model_name=evaluation.candidate.score_model_name,
                score_model_version=evaluation.candidate.score_model_version,
                strategy_name=evaluation.candidate.strategy_name,
                strategy_version=evaluation.candidate.strategy_version,
                strategy_configuration_version=(
                    evaluation.candidate.strategy_configuration_version
                ),
                candidate={
                    **_candidate_payload(evaluation.candidate),
                    "ranking": _json(asdict(evaluation.ranking))
                    if evaluation.ranking is not None
                    else None,
                },
                status=evaluation.status.value,
                explanation=evaluation.explanation,
                rejection_reasons=[item.value for item in evaluation.rejection_reasons],
                valid_window_count=evaluation.valid_window_count,
                confirmed_window_count=evaluation.confirmed_window_count,
                rejected_validation_window_count=evaluation.rejected_validation_window_count,
                rejected_test_window_count=evaluation.rejected_test_window_count,
                training_metrics=_metric(evaluation.training),
                validation_metrics=_metric(evaluation.validation),
                test_metrics=(
                    _metric(evaluation.test) if evaluation.test is not None else None
                ),
                unconditional_benchmark=evaluation.unconditional_benchmark,
            )
        )
        try:
            await session.flush()
        except Exception:
            await session.rollback()
            raise
        for window in evaluation.windows:
            session.add(
                EdgeGuidedWindowRecord(
                    id=uuid4(),
                    candidate_evaluation_id=candidate_id,
                    window_ordinal=window.window_ordinal,
                    cache_key=window.cache_key or None,
                    volatility_threshold=window.volatility_threshold,
                    training_metrics=_metric(window.training),
                    validation_metrics=_metric(window.validation),
                    test_metrics=_metric(window.test) if window.test else None,
                    validation_passed=window.validation_passed,
                    test_confirmed=window.test_confirmed,
                    rejection_reasons=[item.value for item in window.rejection_reasons],
                )
            )
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        reused = await find_reusable_edge_guided_run(session, experiment_key)
        if reused is None:
            raise
        return reused
    except Exception:
        await session.rollback()
        raise
    return EdgeGuidedExecution(
        run_id, experiment_key, False, result, 0, result.evaluated_count, 0
    )


async def _load_result(
    session: AsyncSession, run: EdgeGuidedRunRecord
) -> EdgeGuidedEvaluationResult:
    records = (
        await session.scalars(
            select(EdgeGuidedCandidateRecord)
            .where(EdgeGuidedCandidateRecord.run_id == run.id)
            .order_by(EdgeGuidedCandidateRecord.candidate_fingerprint)
        )
    ).all()
    stage_records = (
        await session.scalars(
            select(EdgeGuidedStageEvidenceRecord)
            .where(EdgeGuidedStageEvidenceRecord.run_id == run.id)
            .order_by(EdgeGuidedStageEvidenceRecord.cache_key)
        )
    ).all()
    candidate_ids = tuple(record.id for record in records)
    window_records = (
        (
            await session.scalars(
                select(EdgeGuidedWindowRecord)
                .where(
                    EdgeGuidedWindowRecord.candidate_evaluation_id.in_(candidate_ids)
                )
                .order_by(
                    EdgeGuidedWindowRecord.candidate_evaluation_id,
                    EdgeGuidedWindowRecord.window_ordinal,
                )
            )
        ).all()
        if candidate_ids
        else []
    )
    windows_by_candidate: dict[UUID, list[EdgeGuidedWindowRecord]] = {}
    for window_record in window_records:
        windows_by_candidate.setdefault(
            window_record.candidate_evaluation_id, []
        ).append(window_record)
    evaluations = []
    for record in records:
        windows = windows_by_candidate.get(record.id, [])
        candidate = _candidate(record.candidate)
        evaluations.append(
            EdgeGuidedCandidateEvaluation(
                candidate,
                _read_metric(record.training_metrics),
                _read_metric(record.validation_metrics),
                _read_metric(record.test_metrics) if record.test_metrics else None,
                tuple(
                    EdgeGuidedWindowEvaluation(
                        item.window_ordinal,
                        item.volatility_threshold,
                        _read_metric(item.training_metrics),
                        _read_metric(item.validation_metrics),
                        _read_metric(item.test_metrics) if item.test_metrics else None,
                        item.validation_passed,
                        item.test_confirmed,
                        tuple(
                            EdgeGuidedRejectionReason(value)
                            for value in item.rejection_reasons
                        ),
                        item.cache_key or "",
                    )
                    for item in windows
                ),
                record.valid_window_count,
                record.confirmed_window_count,
                record.rejected_validation_window_count,
                record.rejected_test_window_count,
                record.unconditional_benchmark,
                EdgeGuidedCandidateStatus(record.status),
                record.explanation,
                tuple(
                    EdgeGuidedRejectionReason(value)
                    for value in record.rejection_reasons
                ),
                _read_ranking(record.candidate.get("ranking")),
            )
        )
    return EdgeGuidedEvaluationResult(
        dataset_fingerprint=run.dataset_fingerprint,
        generated_count=run.generated_count,
        evaluated_count=run.evaluated_count,
        maximum_candidates=run.maximum_candidates,
        windows_available=run.windows_available,
        evaluations=tuple(evaluations),
        outcome=ResearchOutcome(run.research_outcome)
        if run.research_outcome
        else ResearchOutcome.NO_PREDICTIVE_EDGE,
        outcome_explanation=run.outcome_explanation
        or "Legacy research run without an outcome explanation.",
        funnel={key: int(str(value)) for key, value in (run.funnel or {}).items()},
        timings={
            key: Decimal(str(value)) for key, value in (run.timings or {}).items()
        },
        qualification_contract_version=run.qualification_contract_version,
        stage_evidence=tuple(
            EdgeGuidedStageEvidence(
                item.cache_key,
                item.candidate_fingerprint,
                RefinementSegmentName(item.stage),
                item.window_ordinal,
                item.segment_from,
                item.segment_to,
                item.content_fingerprint,
                _read_metric(item.metrics),
            )
            for item in stage_records
        ),
    )


async def _load_valid_result(
    session: AsyncSession, run: EdgeGuidedRunRecord
) -> EdgeGuidedEvaluationResult | None:
    if run.qualification_contract_version != EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION:
        return None
    if run.generated_count < run.evaluated_count:
        return None
    result = await _load_result(session, run)
    if len(result.evaluations) != run.evaluated_count:
        return None
    if run.evaluated_count == 0:
        return result if not result.evaluations else None
    for evaluation in result.evaluations:
        windows = evaluation.windows
        if len(windows) != (
            evaluation.valid_window_count + evaluation.rejected_validation_window_count
        ):
            return None
        ordinals = tuple(item.window_ordinal for item in windows)
        if len(set(ordinals)) != len(ordinals):
            return None
        confirmed = sum(item.test_confirmed for item in windows)
        rejected_validation = sum(not item.validation_passed for item in windows)
        rejected_test = sum(
            item.validation_passed and not item.test_confirmed for item in windows
        )
        if (
            confirmed != evaluation.confirmed_window_count
            or rejected_validation != evaluation.rejected_validation_window_count
            or rejected_test != evaluation.rejected_test_window_count
        ):
            return None
    return result


def _json(value: object) -> dict[str, object]:
    result = json.loads(json.dumps(value, default=str))
    if not isinstance(result, dict):
        raise ValueError("edge-guided value must be an object")
    return result


def _metric(value: EdgeGuidedSegmentEvaluation) -> dict[str, object]:
    return _json(asdict(value))


def _candidate_payload(value: EdgeGuidedCandidate) -> dict[str, object]:
    payload = asdict(value)
    payload["horizon_seconds"] = int(value.horizon.total_seconds())
    payload["decision_interval_seconds"] = int(value.decision_interval.total_seconds())
    payload["cadence_seconds"] = int(value.cadence.total_seconds())
    payload["minimum_holding_seconds"] = int(
        value.minimum_holding_period.total_seconds()
    )
    payload["maximum_holding_seconds"] = int(
        value.maximum_holding_period.total_seconds()
    )
    payload["cooldown_seconds"] = int(value.cooldown.total_seconds())
    del payload["horizon"]
    del payload["decision_interval"]
    del payload["cadence"]
    del payload["minimum_holding_period"]
    del payload["maximum_holding_period"]
    del payload["cooldown"]
    return _json(payload)


def _read_metric(value: dict[str, object]) -> EdgeGuidedSegmentEvaluation:
    return EdgeGuidedSegmentEvaluation(
        RefinementSegmentName(str(value["segment"])),
        Decimal(str(value["total_return"])),
        Decimal(str(value["max_drawdown"])),
        int(str(value["trade_count"])),
        Decimal(str(value["fees_paid"])),
        Decimal(str(value["return_difference_from_production"])),
        Decimal(str(value["return_difference_from_buy_and_hold"])),
    )


def _candidate(value: dict[str, object]) -> EdgeGuidedCandidate:
    source_ids = value.get("source_hypothesis_ids", [])
    if not isinstance(source_ids, list):
        source_ids = []
    return EdgeGuidedCandidate(
        str(value["candidate_fingerprint"]),
        str(value["source_edge_fingerprint"]),
        str(value["source_hypothesis_id"]),
        str(value["predictor_name"]),
        int(str(value["predictor_version"])),
        int(str(value["hypothesis_contract_version"])),
        Condition(str(value["condition"])),
        Decimal(str(value["lower_bound"]))
        if value["lower_bound"] is not None
        else None,
        Decimal(str(value["upper_bound"]))
        if value["upper_bound"] is not None
        else None,
        SuggestedAction(str(value["suggested_action"])),
        timedelta(seconds=int(str(value["horizon_seconds"]))),
        timedelta(seconds=int(str(value["decision_interval_seconds"]))),
        DirectionMapping(str(value["direction_mapping"])),
        str(value["score_model_name"]),
        int(str(value["score_model_version"])),
        str(value["strategy_name"]),
        int(str(value["strategy_version"])),
        int(str(value["strategy_configuration_version"])),
        str(value["account_currency"]),
        Decimal(str(value["fee_rate"])),
        Decimal(str(value["slippage_rate"])),
        CandidateFamily(str(value.get("candidate_family", "exit_or_avoid_overlay"))),
        EntryRule(str(value.get("entry_rule", "production_baseline"))),
        ExitRule(str(value.get("exit_rule", "defensive_overlay"))),
        timedelta(
            seconds=int(
                str(value.get("cadence_seconds", value["decision_interval_seconds"]))
            )
        ),
        timedelta(seconds=int(str(value.get("minimum_holding_seconds", 0)))),
        timedelta(seconds=int(str(value.get("maximum_holding_seconds", 0)))),
        timedelta(seconds=int(str(value.get("cooldown_seconds", 0)))),
        Decimal(str(value.get("target_allocation_percentage", "15"))),
        str(value["exit_predictor_name"])
        if value.get("exit_predictor_name") is not None
        else None,
        Condition(str(value["exit_condition"]))
        if value.get("exit_condition") is not None
        else None,
        Decimal(str(value["exit_lower_bound"]))
        if value.get("exit_lower_bound") is not None
        else None,
        Decimal(str(value["exit_upper_bound"]))
        if value.get("exit_upper_bound") is not None
        else None,
        tuple(str(item) for item in source_ids),
        int(str(value.get("generation_contract_version", 1))),
        int(str(value.get("qualification_contract_version", 1))),
    )


def _read_ranking(value: object) -> EdgeGuidedRanking | None:
    if not isinstance(value, dict):
        return None
    return EdgeGuidedRanking(
        Decimal(str(value["median_validation_return"])),
        Decimal(str(value["worst_validation_return"])),
        Decimal(str(value["maximum_drawdown"])),
        Decimal(str(value["return_over_drawdown"]))
        if value.get("return_over_drawdown") is not None
        else None,
        int(str(value["trade_count"])),
        Decimal(str(value["total_fees"])),
        Decimal(str(value["cost_efficiency"]))
        if value.get("cost_efficiency") is not None
        else None,
        Decimal(str(value["positive_window_ratio"])),
        int(str(value["positive_window_count"])),
        str(value["candidate_fingerprint"]),
    )
