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
    predictive_edge_run_id: UUID | None = None


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
            item.segment_to×¿w¶‰žËkºwµç@€€€€€€€€€€€€€Í½É•}µ½‘•±}¹…µ”õ•Ù…±Õ…Ñ¥½¸¹…¹‘¥‘…Ñ”¹Í½É•}µ½‘•±}¹…µ”°(€€€€€€€€€€€€€€€Í½É•}µ½‘•±}Ù•ÉÍ¥½¸õ•Ù…±Õ…Ñ¥½¸¹…¹‘¥‘…Ñ”¹Í½É•}µ½‘•±}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€ÍÑÉ…Ñ•å}¹…µ”õ•Ù…±Õ…Ñ¥½¸¹…¹‘¥‘…Ñ”¹ÍÑÉ…Ñ•å}¹…µ”°(€€€€€€€€€€€€€€€ÍÑÉ…Ñ•å}Ù•ÉÍ¥½¸õ•Ù…±Õ…Ñ¥½¸¹…¹‘¥‘…Ñ”¹ÍÑÉ…Ñ•å}Ù•ÉÍ¥½¸°(€€€€€€€€€€€€€€€ÍÑÉ…Ñ•å}½¹™¥ÕÉ…Ñ¥½¹}Ù•ÉÍ¥½¸ô (€€€€€€€€€€€€€€€€€€€•Ù…±Õ…Ñ¥½¸¹…¹‘¥‘…Ñ”¹ÍÑÉ…Ñ•å}½¹™¥ÕÉ…Ñ¥½¹}Ù•ÉÍ¥½¸(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”õì(€€€€€€€€€€€€€€€€€€€€¨©}…¹‘¥‘…Ñ•}Á…å±½…¡•Ù…±Õ…Ñ¥½¸¹…¹‘¥‘…Ñ”¤°(€€€€€€€€€€€€€€€€€€€€‰É…¹­¥¹œˆè}©Í½¸¡…Í‘¥Ð¡•Ù…±Õ…Ñ¥½¸¹É…¹­¥¹œ¤¤(€€€€€€€€€€€€€€€€€€€¥˜•Ù…±Õ…Ñ¥½¸¹É…¹­¥¹œ¥Ì¹½Ð9½¹”(€€€€€€€€€€€€€€€€€€€•±Í”9½¹”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€€€€€ÍÑ…ÑÕÌõ•Ù…±Õ…Ñ¥½¸¹ÍÑ…ÑÕÌ¹Ù…±Õ”°(€€€€€€€€€€€€€€€•áÁ±…¹…Ñ¥½¸õ•Ù…±Õ…Ñ¥½¸¹•áÁ±…¹…Ñ¥½¸°(€€€€€€€€€€€€€€€É•©•Ñ¥½¹}É•…Í½¹Ìõm¥Ñ•´¹Ù…±Õ”™½È¥Ñ•´¥¸•Ù…±Õ…Ñ¥½¸¹É•©•Ñ¥½¹}É•…Í½¹Ít°(€€€€€€€€€€€€€€€Ù…±¥‘}Ý¥¹‘½Ý}½Õ¹Ðõ•Ù…±Õ…Ñ¥½¸¹Ù…±¥‘}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€½¹™¥Éµ•‘}Ý¥¹‘½Ý}½Õ¹Ðõ•Ù…±Õ…Ñ¥½¸¹½¹™¥Éµ•‘}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¹}Ý¥¹‘½Ý}½Õ¹Ðõ•Ù…±Õ…Ñ¥½¸¹É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¹}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€É•©•Ñ•‘}Ñ•ÍÑ}Ý¥¹‘½Ý}½Õ¹Ðõ•Ù…±Õ…Ñ¥½¸¹É•©•Ñ•‘}Ñ•ÍÑ}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€ÑÉ…¥¹¥¹}µ•ÑÉ¥Ìõ}µ•ÑÉ¥Œ¡•Ù…±Õ…Ñ¥½¸¹ÑÉ…¥¹¥¹œ¤°(€€€€€€€€€€€€€€€Ù…±¥‘…Ñ¥½¹}µ•ÑÉ¥Ìõ}µ•ÑÉ¥Œ¡•Ù…±Õ…Ñ¥½¸¹Ù…±¥‘…Ñ¥½¸¤°(€€€€€€€€€€€€€€€Ñ•ÍÑ}µ•ÑÉ¥Ìô (€€€€€€€€€€€€€€€€€€€}µ•ÑÉ¥Œ¡•Ù…±Õ…Ñ¥½¸¹Ñ•ÍÐ¤¥˜•Ù…±Õ…Ñ¥½¸¹Ñ•ÍÐ¥Ì¹½Ð9½¹”•±Í”9½¹”(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€Õ¹½¹‘¥Ñ¥½¹…±}‰•¹¡µ…É¬õ•Ù…±Õ…Ñ¥½¸¹Õ¹½¹‘¥Ñ¥½¹…±}‰•¹¡µ…É¬°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€€€€€ÑÉäè(€€€€€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹™±ÕÍ  ¤(€€€€€€€•á•ÁÐá•ÁÑ¥½¸è(€€€€€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹É½±±‰…¬ ¤(€€€€€€€€€€€É…¥Í”(€€€€€€€™½ÈÝ¥¹‘½Ü¥¸•Ù…±Õ…Ñ¥½¸¹Ý¥¹‘½ÝÌè(€€€€€€€€€€€Í•ÍÍ¥½¸¹…‘ (€€€€€€€€€€€€€€€‘•Õ¥‘•‘]¥¹‘½ÝI•½É (€€€€€€€€€€€€€€€€€€€¥õÕÕ¥Ð ¤°(€€€€€€€€€€€€€€€€€€€…¹‘¥‘…Ñ•}•Ù…±Õ…Ñ¥½¹}¥õ…¹‘¥‘…Ñ•}¥°(€€€€€€€€€€€€€€€€€€€Ý¥¹‘½Ý}½É‘¥¹…°õÝ¥¹‘½Ü¹Ý¥¹‘½Ý}½É‘¥¹…°°(€€€€€€€€€€€€€€€€€€€…¡•}­•äõÝ¥¹‘½Ü¹…¡•}­•ä½È9½¹”°(€€€€€€€€€€€€€€€€€€€Ù½±…Ñ¥±¥Ñå}Ñ¡É•Í¡½±õÝ¥¹‘½Ü¹Ù½±…Ñ¥±¥Ñå}Ñ¡É•Í¡½±°(€€€€€€€€€€€€€€€€€€€ÑÉ…¥¹¥¹}µ•ÑÉ¥Ìõ}µ•ÑÉ¥Œ¡Ý¥¹‘½Ü¹ÑÉ…¥¹¥¹œ¤°(€€€€€€€€€€€€€€€€€€€Ù…±¥‘…Ñ¥½¹}µ•ÑÉ¥Ìõ}µ•ÑÉ¥Œ¡Ý¥¹‘½Ü¹Ù…±¥‘…Ñ¥½¸¤°(€€€€€€€€€€€€€€€€€€€Ñ•ÍÑ}µ•ÑÉ¥Ìõ}µ•ÑÉ¥Œ¡Ý¥¹‘½Ü¹Ñ•ÍÐ¤¥˜Ý¥¹‘½Ü¹Ñ•ÍÐ•±Í”9½¹”°(€€€€€€€€€€€€€€€€€€€Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•õÝ¥¹‘½Ü¹Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•°(€€€€€€€€€€€€€€€€€€€Ñ•ÍÑ}½¹™¥Éµ•õÝ¥¹‘½Ü¹Ñ•ÍÑ}½¹™¥Éµ•°(€€€€€€€€€€€€€€€€€€€É•©•Ñ¥½¹}É•…Í½¹Ìõm¥Ñ•´¹Ù…±Õ”™½È¥Ñ•´¥¸Ý¥¹‘½Ü¹É•©•Ñ¥½¹}É•…Í½¹Ít°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€ÑÉäè(€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹½µµ¥Ð ¤(€€€•á•ÁÐ%¹Ñ•É¥ÑåÉÉ½Èè(€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹É½±±‰…¬ ¤(€€€€€€€É•ÕÍ•€ô…Ý…¥Ð™¥¹‘}É•ÕÍ…‰±•}•‘•}Õ¥‘•‘}ÉÕ¸¡Í•ÍÍ¥½¸°•áÁ•É¥µ•¹Ñ}­•ä¤(€€€€€€€¥˜É•ÕÍ•¥Ì9½¹”è(€€€€€€€€€€€É…¥Í”(€€€€€€€É•ÑÕÉ¸É•ÕÍ•(€€€•á•ÁÐá•ÁÑ¥½¸è(€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹É½±±‰…¬ ¤(€€€€€€€É…¥Í”(€€€É•ÑÕÉ¸‘•Õ¥‘•‘á•ÕÑ¥½¸ (€€€€€€€ÉÕ¹}¥°(€€€€€€€•áÁ•É¥µ•¹Ñ}­•ä°(€€€€€€€…±Í”°(€€€€€€€É•ÍÕ±Ð°(€€€€€€€€À°(€€€€€€€É•ÍÕ±Ð¹•Ù…±Õ…Ñ•‘}½Õ¹Ð°(€€€€€€€€À°(€€€€€€€ÁÉ•‘¥Ñ¥Ù•}•‘•}ÉÕ¹}¥°(€€€€¤(()…Íå¹Œ‘•˜}±½…‘}É•ÍÕ±Ð (€€€Í•ÍÍ¥½¸èÍå¹M•ÍÍ¥½¸°ÉÕ¸è‘•Õ¥‘•‘IÕ¹I•½É(¤€´ø‘•Õ¥‘•‘Ù…±Õ…Ñ¥½¹I•ÍÕ±Ðè(€€€É•½É‘Ì€ô€ (€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹Í…±…ÉÌ (€€€€€€€€€€€Í•±•Ð¡‘•Õ¥‘•‘…¹‘¥‘…Ñ•I•½É¤(€€€€€€€€€€€€¹Ý¡•É”¡‘•Õ¥‘•‘…¹‘¥‘…Ñ•I•½É¹ÉÕ¹}¥€ôôÉÕ¸¹¥¤(€€€€€€€€€€€€¹½É‘•É}‰ä¡‘•Õ¥‘•‘…¹‘¥‘…Ñ•I•½É¹…¹‘¥‘…Ñ•}™¥¹•ÉÁÉ¥¹Ð¤(€€€€€€€€¤(€€€€¤¹…±° ¤(€€€ÍÑ…•}É•½É‘Ì€ô€ (€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹Í…±…ÉÌ (€€€€€€€€€€€Í•±•Ð¡‘•Õ¥‘•‘MÑ…•Ù¥‘•¹•I•½É¤(€€€€€€€€€€€€¹Ý¡•É”¡‘•Õ¥‘•‘MÑ…•Ù¥‘•¹•I•½É¹ÉÕ¹}¥€ôôÉÕ¸¹¥¤(€€€€€€€€€€€€¹½É‘•É}‰ä¡‘•Õ¥‘•‘MÑ…•Ù¥‘•¹•I•½É¹…¡•}­•ä¤(€€€€€€€€¤(€€€€¤¹…±° ¤(€€€…¹‘¥‘…Ñ•}¥‘Ì€ôÑÕÁ±”¡É•½É¹¥™½ÈÉ•½É¥¸É•½É‘Ì¤(€€€Ý¥¹‘½Ý}É•½É‘Ì€ô€ (€€€€€€€€ (€€€€€€€€€€€…Ý…¥ÐÍ•ÍÍ¥½¸¹Í…±…ÉÌ (€€€€€€€€€€€€€€€Í•±•Ð¡‘•Õ¥‘•‘]¥¹‘½ÝI•½É¤(€€€€€€€€€€€€€€€€¹Ý¡•É” (€€€€€€€€€€€€€€€€€€€‘•Õ¥‘•‘]¥¹‘½ÝI•½É¹…¹‘¥‘…Ñ•}•Ù…±Õ…Ñ¥½¹}¥¹¥¹|¡…¹‘¥‘…Ñ•}¥‘Ì¤(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€¹½É‘•É}‰ä (€€€€€€€€€€€€€€€€€€€‘•Õ¥‘•‘]¥¹‘½ÝI•½É¹…¹‘¥‘…Ñ•}•Ù…±Õ…Ñ¥½¹}¥°(€€€€€€€€€€€€€€€€€€€‘•Õ¥‘•‘]¥¹‘½ÝI•½É¹Ý¥¹‘½Ý}½É‘¥¹…°°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€€€€€€¤¹…±° ¤(€€€€€€€¥˜…¹‘¥‘…Ñ•}¥‘Ì(€€€€€€€•±Í”mt(€€€€¤(€€€Ý¥¹‘½ÝÍ}‰å}…¹‘¥‘…Ñ”è‘¥ÑmUU%°±¥ÍÑm‘•Õ¥‘•‘]¥¹‘½ÝI•½É‘ut€ôíô(€€€™½ÈÝ¥¹‘½Ý}É•½É¥¸Ý¥¹‘½Ý}É•½É‘Ìè(€€€€€€€Ý¥¹‘½ÝÍ}‰å}…¹‘¥‘…Ñ”¹Í•Ñ‘•™…Õ±Ð (€€€€€€€€€€€Ý¥¹‘½Ý}É•½É¹…¹‘¥‘…Ñ•}•Ù…±Õ…Ñ¥½¹}¥°mt(€€€€€€€€¤¹…ÁÁ•¹¡Ý¥¹‘½Ý}É•½É¤(€€€•Ù…±Õ…Ñ¥½¹Ì€ômt(€€€™½ÈÉ•½É¥¸É•½É‘Ìè(€€€€€€€Ý¥¹‘½ÝÌ€ôÝ¥¹‘½ÝÍ}‰å}…¹‘¥‘…Ñ”¹•Ð¡É•½É¹¥°mt¤(€€€€€€€…¹‘¥‘…Ñ”€ô}…¹‘¥‘…Ñ”¡É•½É¹…¹‘¥‘…Ñ”¤(€€€€€€€•Ù…±Õ…Ñ¥½¹Ì¹…ÁÁ•¹ (€€€€€€€€€€€‘•Õ¥‘•‘…¹‘¥‘…Ñ•Ù…±Õ…Ñ¥½¸ (€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”°(€€€€€€€€€€€€€€€}É•…‘}µ•ÑÉ¥Œ¡É•½É¹ÑÉ…¥¹¥¹}µ•ÑÉ¥Ì¤°(€€€€€€€€€€€€€€€}É•…‘}µ•ÑÉ¥Œ¡É•½É¹Ù…±¥‘…Ñ¥½¹}µ•ÑÉ¥Ì¤°(€€€€€€€€€€€€€€€}É•…‘}µ•ÑÉ¥Œ¡É•½É¹Ñ•ÍÑ}µ•ÑÉ¥Ì¤¥˜É•½É¹Ñ•ÍÑ}µ•ÑÉ¥Ì•±Í”9½¹”°(€€€€€€€€€€€€€€€ÑÕÁ±” (€€€€€€€€€€€€€€€€€€€‘•Õ¥‘•‘]¥¹‘½ÝÙ…±Õ…Ñ¥½¸ (€€€€€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Ý¥¹‘½Ý}½É‘¥¹…°°(€€€€€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Ù½±…Ñ¥±¥Ñå}Ñ¡É•Í¡½±°(€€€€€€€€€€€€€€€€€€€€€€€}É•…‘}µ•ÑÉ¥Œ¡¥Ñ•´¹ÑÉ…¥¹¥¹}µ•ÑÉ¥Ì¤°(€€€€€€€€€€€€€€€€€€€€€€€}É•…‘}µ•ÑÉ¥Œ¡¥Ñ•´¹Ù…±¥‘…Ñ¥½¹}µ•ÑÉ¥Ì¤°(€€€€€€€€€€€€€€€€€€€€€€€}É•…‘}µ•ÑÉ¥Œ¡¥Ñ•´¹Ñ•ÍÑ}µ•ÑÉ¥Ì¤¥˜¥Ñ•´¹Ñ•ÍÑ}µ•ÑÉ¥Ì•±Í”9½¹”°(€€€€€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•°(€€€€€€€€€€€€€€€€€€€€€€€¥Ñ•´¹Ñ•ÍÑ}½¹™¥Éµ•°(€€€€€€€€€€€€€€€€€€€€€€€ÑÕÁ±” (€€€€€€€€€€€€€€€€€€€€€€€€€€€‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¡Ù…±Õ”¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€™½ÈÙ…±Õ”¥¸¥Ñ•´¹É•©•Ñ¥½¹}É•…Í½¹Ì(€€€€€€€€€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€€€€€€€€¥Ñ•´¹…¡•}­•ä½È€ˆˆ°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€™½È¥Ñ•´¥¸Ý¥¹‘½ÝÌ(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€É•½É¹Ù…±¥‘}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€É•½É¹½¹™¥Éµ•‘}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€É•½É¹É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¹}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€É•½É¹É•©•Ñ•‘}Ñ•ÍÑ}Ý¥¹‘½Ý}½Õ¹Ð°(€€€€€€€€€€€€€€€É•½É¹Õ¹½¹‘¥Ñ¥½¹…±}‰•¹¡µ…É¬°(€€€€€€€€€€€€€€€‘•Õ¥‘•‘…¹‘¥‘…Ñ•MÑ…ÑÕÌ¡É•½É¹ÍÑ…ÑÕÌ¤°(€€€€€€€€€€€€€€€É•½É¹•áÁ±…¹…Ñ¥½¸°(€€€€€€€€€€€€€€€ÑÕÁ±” (€€€€€€€€€€€€€€€€€€€‘•Õ¥‘•‘I•©•Ñ¥½¹I•…Í½¸¡Ù…±Õ”¤(€€€€€€€€€€€€€€€€€€€™½ÈÙ…±Õ”¥¸É•½É¹É•©•Ñ¥½¹}É•…Í½¹Ì(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€}É•…‘}É…¹­¥¹œ¡É•½É¹…¹‘¥‘…Ñ”¹•Ð ‰É…¹­¥¹œˆ¤¤°(€€€€€€€€€€€€¤(€€€€€€€€¤(€€€É•ÑÕÉ¸‘•Õ¥‘•‘Ù…±Õ…Ñ¥½¹I•ÍÕ±Ð (€€€€€€€‘…Ñ…Í•Ñ}™¥¹•ÉÁÉ¥¹ÐõÉÕ¸¹‘…Ñ…Í•Ñ}™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€•¹•É…Ñ•‘}½Õ¹ÐõÉÕ¸¹•¹•É…Ñ•‘}½Õ¹Ð°(€€€€€€€•Ù…±Õ…Ñ•‘}½Õ¹ÐõÉÕ¸¹•Ù…±Õ…Ñ•‘}½Õ¹Ð°(€€€€€€€µ…á¥µÕµ}…¹‘¥‘…Ñ•ÌõÉÕ¸¹µ…á¥µÕµ}…¹‘¥‘…Ñ•Ì°(€€€€€€€Ý¥¹‘½ÝÍ}…Ù…¥±…‰±”õÉÕ¸¹Ý¥¹‘½ÝÍ}…Ù…¥±…‰±”°(€€€€€€€•Ù…±Õ…Ñ¥½¹ÌõÑÕÁ±”¡•Ù…±Õ…Ñ¥½¹Ì¤°(€€€€€€€½ÕÑ½µ”õI•Í•…É¡=ÕÑ½µ”¡ÉÕ¸¹É•Í•…É¡}½ÕÑ½µ”¤(€€€€€€€¥˜ÉÕ¸¹É•Í•…É¡}½ÕÑ½µ”(€€€€€€€•±Í”I•Í•…É¡=ÕÑ½µ”¹9=}AI%Q%Y}°(€€€€€€€½ÕÑ½µ•}•áÁ±…¹…Ñ¥½¸õÉÕ¸¹½ÕÑ½µ•}•áÁ±…¹…Ñ¥½¸(€€€€€€€½È€‰1•…äÉ•Í•…É ÉÕ¸Ý¥Ñ¡½ÕÐ…¸½ÕÑ½µ”•áÁ±…¹…Ñ¥½¸¸ˆ°(€€€€€€€™Õ¹¹•°õí­•äè¥¹Ð¡ÍÑÈ¡Ù…±Õ”¤¤™½È­•ä°Ù…±Õ”¥¸€¡ÉÕ¸¹™Õ¹¹•°½Èíô¤¹¥Ñ•µÌ ¥ô°(€€€€€€€Ñ¥µ¥¹Ìõì(€€€€€€€€€€€­•äè•¥µ…°¡ÍÑÈ¡Ù…±Õ”¤¤™½È­•ä°Ù…±Õ”¥¸€¡ÉÕ¸¹Ñ¥µ¥¹Ì½Èíô¤¹¥Ñ•µÌ ¤(€€€€€€€ô°(€€€€€€€ÅÕ…±¥™¥…Ñ¥½¹}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸õÉÕ¸¹ÅÕ…±¥™¥…Ñ¥½¹}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸°(€€€€€€€ÍÑ…•}•Ù¥‘•¹”õÑÕÁ±” (€€€€€€€€€€€‘•Õ¥‘•‘MÑ…•Ù¥‘•¹” (€€€€€€€€€€€€€€€¥Ñ•´¹…¡•}­•ä°(€€€€€€€€€€€€€€€¥Ñ•´¹…¹‘¥‘…Ñ•}™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¡¥Ñ•´¹ÍÑ…”¤°(€€€€€€€€€€€€€€€¥Ñ•´¹Ý¥¹‘½Ý}½É‘¥¹…°°(€€€€€€€€€€€€€€€¥Ñ•´¹Í•µ•¹Ñ}™É½´°(€€€€€€€€€€€€€€€¥Ñ•´¹Í•µ•¹Ñ}Ñ¼°(€€€€€€€€€€€€€€€¥Ñ•´¹½¹Ñ•¹Ñ}™¥¹•ÉÁÉ¥¹Ð°(€€€€€€€€€€€€€€€}É•…‘}µ•ÑÉ¥Œ¡¥Ñ•´¹µ•ÑÉ¥Ì¤°(€€€€€€€€€€€€¤(€€€€€€€€€€€™½È¥Ñ•´¥¸ÍÑ…•}É•½É‘Ì(€€€€€€€€¤°(€€€€¤(()…Íå¹Œ‘•˜}±½…‘}Ù…±¥‘}É•ÍÕ±Ð (€€€Í•ÍÍ¥½¸èÍå¹M•ÍÍ¥½¸°ÉÕ¸è‘•Õ¥‘•‘IÕ¹I•½É(¤€´ø‘•Õ¥‘•‘Ù…±Õ…Ñ¥½¹I•ÍÕ±Ðð9½¹”è(€€€¥˜ÉÕ¸¹ÅÕ…±¥™¥…Ñ¥½¹}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸€„ô}U%}EU1%%Q%=9}=9QIQ}YIM%=8è(€€€€€€€É•ÑÕÉ¸9½¹”(€€€¥˜ÉÕ¸¹•¹•É…Ñ•‘}½Õ¹Ð€ðÉÕ¸¹•Ù…±Õ…Ñ•‘}½Õ¹Ðè(€€€€€€€É•ÑÕÉ¸9½¹”(€€€É•ÍÕ±Ð€ô…Ý…¥Ð}±½…‘}É•ÍÕ±Ð¡Í•ÍÍ¥½¸°ÉÕ¸¤(€€€¥˜±•¸¡É•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹Ì¤€„ôÉÕ¸¹•Ù…±Õ…Ñ•‘}½Õ¹Ðè(€€€€€€€É•ÑÕÉ¸9½¹”(€€€¥˜ÉÕ¸¹•Ù…±Õ…Ñ•‘}½Õ¹Ð€ôô€Àè(€€€€€€€É•ÑÕÉ¸É•ÍÕ±Ð¥˜¹½ÐÉ•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹Ì•±Í”9½¹”(€€€™½È•Ù…±Õ…Ñ¥½¸¥¸É•ÍÕ±Ð¹•Ù…±Õ…Ñ¥½¹Ìè(€€€€€€€Ý¥¹‘½ÝÌ€ô•Ù…±Õ…Ñ¥½¸¹Ý¥¹‘½ÝÌ(€€€€€€€¥˜±•¸¡Ý¥¹‘½ÝÌ¤€„ô€ (€€€€€€€€€€€•Ù…±Õ…Ñ¥½¸¹Ù…±¥‘}Ý¥¹‘½Ý}½Õ¹Ð€¬•Ù…±Õ…Ñ¥½¸¹É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¹}Ý¥¹‘½Ý}½Õ¹Ð(€€€€€€€€¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”(€€€€€€€½É‘¥¹…±Ì€ôÑÕÁ±”¡¥Ñ•´¹Ý¥¹‘½Ý}½É‘¥¹…°™½È¥Ñ•´¥¸Ý¥¹‘½ÝÌ¤(€€€€€€€¥˜±•¸¡Í•Ð¡½É‘¥¹…±Ì¤¤€„ô±•¸¡½É‘¥¹…±Ì¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”(€€€€€€€½¹™¥Éµ•€ôÍÕ´¡¥Ñ•´¹Ñ•ÍÑ}½¹™¥Éµ•™½È¥Ñ•´¥¸Ý¥¹‘½ÝÌ¤(€€€€€€€É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¸€ôÍÕ´¡¹½Ð¥Ñ•´¹Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•™½È¥Ñ•´¥¸Ý¥¹‘½ÝÌ¤(€€€€€€€É•©•Ñ•‘}Ñ•ÍÐ€ôÍÕ´ (€€€€€€€€€€€¥Ñ•´¹Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•…¹¹½Ð¥Ñ•´¹Ñ•ÍÑ}½¹™¥Éµ•™½È¥Ñ•´¥¸Ý¥¹‘½ÝÌ(€€€€€€€€¤(€€€€€€€¥˜€ (€€€€€€€€€€€½¹™¥Éµ•€„ô•Ù…±Õ…Ñ¥½¸¹½¹™¥Éµ•‘}Ý¥¹‘½Ý}½Õ¹Ð(€€€€€€€€€€€½ÈÉ•©•Ñ•‘}Ù…±¥‘…Ñ¥½¸€„ô•Ù…±Õ…Ñ¥½¸¹É•©•Ñ•‘}Ù…±¥‘…Ñ¥½¹}Ý¥¹‘½Ý}½Õ¹Ð(€€€€€€€€€€€½ÈÉ•©•Ñ•‘}Ñ•ÍÐ€„ô•Ù…±Õ…Ñ¥½¸¹É•©•Ñ•‘}Ñ•ÍÑ}Ý¥¹‘½Ý}½Õ¹Ð(€€€€€€€€¤è(€€€€€€€€€€€É•ÑÕÉ¸9½¹”(€€€É•ÑÕÉ¸É•ÍÕ±Ð(()‘•˜}©Í½¸¡Ù…±Õ”è½‰©•Ð¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€É•ÍÕ±Ð€ô©Í½¸¹±½…‘Ì¡©Í½¸¹‘ÕµÁÌ¡Ù…±Õ”°‘•™…Õ±ÐõÍÑÈ¤¤(€€€¥˜¹½Ð¥Í¥¹ÍÑ…¹”¡É•ÍÕ±Ð°‘¥Ð¤è(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰•‘”µÕ¥‘•Ù…±Õ”µÕÍÐ‰”…¸½‰©•Ðˆ¤(€€€É•ÑÕÉ¸É•ÍÕ±Ð(()‘•˜}µ•ÑÉ¥Œ¡Ù…±Õ”è‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€É•ÑÕÉ¸}©Í½¸¡…Í‘¥Ð¡Ù…±Õ”¤¤(()‘•˜}…¹‘¥‘…Ñ•}Á…å±½…¡Ù…±Õ”è‘•Õ¥‘•‘…¹‘¥‘…Ñ”¤€´ø‘¥ÑmÍÑÈ°½‰©•Ñtè(€€€Á…å±½…€ô…Í‘¥Ð¡Ù…±Õ”¤(€€€Á…å±½…‘l‰¡½É¥é½¹}Í•½¹‘Ì‰t€ô¥¹Ð¡Ù…±Õ”¹¡½É¥é½¸¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤(€€€Á…å±½…‘l‰‘•¥Í¥½¹}¥¹Ñ•ÉÙ…±}Í•½¹‘Ì‰t€ô¥¹Ð¡Ù…±Õ”¹‘•¥Í¥½¹}¥¹Ñ•ÉÙ…°¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤(€€€Á…å±½…‘l‰…‘•¹•}Í•½¹‘Ì‰t€ô¥¹Ð¡Ù…±Õ”¹…‘•¹”¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤(€€€Á…å±½…‘l‰µ¥¹¥µÕµ}¡½±‘¥¹}Í•½¹‘Ì‰t€ô¥¹Ð (€€€€€€€Ù…±Õ”¹µ¥¹¥µÕµ}¡½±‘¥¹}Á•É¥½¹Ñ½Ñ…±}Í•½¹‘Ì ¤(€€€€¤(€€€Á…å±½…‘l‰µ…á¥µÕµ}¡½±‘¥¹}Í•½¹‘Ì‰t€ô¥¹Ð (€€€€€€€Ù…±Õ”¹µ…á¥µÕµ}¡½±‘¥¹}Á•É¥½¹Ñ½Ñ…±}Í•½¹‘Ì ¤(€€€€¤(€€€Á…å±½…‘l‰½½±‘½Ý¹}Í•½¹‘Ì‰t€ô¥¹Ð¡Ù…±Õ”¹½½±‘½Ý¸¹Ñ½Ñ…±}Í•½¹‘Ì ¤¤(€€€‘•°Á…å±½…‘l‰¡½É¥é½¸‰t(€€€‘•°Á…å±½…‘l‰‘•¥Í¥½¹}¥¹Ñ•ÉÙ…°‰t(€€€‘•°Á…å±½…‘l‰…‘•¹”‰t(€€€‘•°Á…å±½…‘l‰µ¥¹¥µÕµ}¡½±‘¥¹}Á•É¥½‰t(€€€‘•°Á…å±½…‘l‰µ…á¥µÕµ}¡½±‘¥¹}Á•É¥½‰t(€€€‘•°Á…å±½…‘l‰½½±‘½Ý¸‰t(€€€É•ÑÕÉ¸}©Í½¸¡Á…å±½…¤(()‘•˜}É•…‘}µ•ÑÉ¥Œ¡Ù…±Õ”è‘¥ÑmÍÑÈ°½‰©•Ñt¤€´ø‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸è(€€€É•ÑÕÉ¸‘•Õ¥‘•‘M•µ•¹ÑÙ…±Õ…Ñ¥½¸ (€€€€€€€I•™¥¹•µ•¹ÑM•µ•¹Ñ9…µ”¡ÍÑÈ¡Ù…±Õ•l‰Í•µ•¹Ð‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰Ñ½Ñ…±}É•ÑÕÉ¸‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰µ…á}‘É…Ý‘½Ý¸‰t¤¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰ÑÉ…‘•}½Õ¹Ð‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰™••Í}Á…¥‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰É•ÑÕÉ¹}‘¥™™•É•¹•}™É½µ}ÁÉ½‘ÕÑ¥½¸‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰É•ÑÕÉ¹}‘¥™™•É•¹•}™É½µ}‰Õå}…¹‘}¡½±‰t¤¤°(€€€€¤(()‘•˜}…¹‘¥‘…Ñ”¡Ù…±Õ”è‘¥ÑmÍÑÈ°½‰©•Ñt¤€´ø‘•Õ¥‘•‘…¹‘¥‘…Ñ”è(€€€Í½ÕÉ•}¥‘Ì€ôÙ…±Õ”¹•Ð ‰Í½ÕÉ•}¡åÁ½Ñ¡•Í¥Í}¥‘Ìˆ°mt¤(€€€¥˜¹½Ð¥Í¥¹ÍÑ…¹”¡Í½ÕÉ•}¥‘Ì°±¥ÍÐ¤è(€€€€€€€Í½ÕÉ•}¥‘Ì€ômt(€€€É•ÑÕÉ¸‘•Õ¥‘•‘…¹‘¥‘…Ñ” (€€€€€€€ÍÑÈ¡Ù…±Õ•l‰…¹‘¥‘…Ñ•}™¥¹•ÉÁÉ¥¹Ð‰t¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰Í½ÕÉ•}•‘•}™¥¹•ÉÁÉ¥¹Ð‰t¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰Í½ÕÉ•}¡åÁ½Ñ¡•Í¥Í}¥‰t¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰ÁÉ•‘¥Ñ½É}¹…µ”‰t¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸‰t¤¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰¡åÁ½Ñ¡•Í¥Í}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸‰t¤¤°(€€€€€€€½¹‘¥Ñ¥½¸¡ÍÑÈ¡Ù…±Õ•l‰½¹‘¥Ñ¥½¸‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰±½Ý•É}‰½Õ¹‰t¤¤(€€€€€€€¥˜Ù…±Õ•l‰±½Ý•É}‰½Õ¹‰t¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰ÕÁÁ•É}‰½Õ¹‰t¤¤(€€€€€€€¥˜Ù…±Õ•l‰ÕÁÁ•É}‰½Õ¹‰t¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€MÕ•ÍÑ•‘Ñ¥½¸¡ÍÑÈ¡Ù…±Õ•l‰ÍÕ•ÍÑ•‘}…Ñ¥½¸‰t¤¤°(€€€€€€€Ñ¥µ•‘•±Ñ„¡Í•½¹‘Ìõ¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰¡½É¥é½¹}Í•½¹‘Ì‰t¤¤¤°(€€€€€€€Ñ¥µ•‘•±Ñ„¡Í•½¹‘Ìõ¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰‘•¥Í¥½¹}¥¹Ñ•ÉÙ…±}Í•½¹‘Ì‰t¤¤¤°(€€€€€€€¥É•Ñ¥½¹5…ÁÁ¥¹œ¡ÍÑÈ¡Ù…±Õ•l‰‘¥É•Ñ¥½¹}µ…ÁÁ¥¹œ‰t¤¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰Í½É•}µ½‘•±}¹…µ”‰t¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰Í½É•}µ½‘•±}Ù•ÉÍ¥½¸‰t¤¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰ÍÑÉ…Ñ•å}¹…µ”‰t¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰ÍÑÉ…Ñ•å}Ù•ÉÍ¥½¸‰t¤¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰ÍÑÉ…Ñ•å}½¹™¥ÕÉ…Ñ¥½¹}Ù•ÉÍ¥½¸‰t¤¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰…½Õ¹Ñ}ÕÉÉ•¹ä‰t¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰™••}É…Ñ”‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰Í±¥ÁÁ…•}É…Ñ”‰t¤¤°(€€€€€€€…¹‘¥‘…Ñ•…µ¥±ä¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰…¹‘¥‘…Ñ•}™…µ¥±äˆ°€‰•á¥Ñ}½É}…Ù½¥‘}½Ù•É±…äˆ¤¤¤°(€€€€€€€¹ÑÉåIÕ±”¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰•¹ÑÉå}ÉÕ±”ˆ°€‰ÁÉ½‘ÕÑ¥½¹}‰…Í•±¥¹”ˆ¤¤¤°(€€€€€€€á¥ÑIÕ±”¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰•á¥Ñ}ÉÕ±”ˆ°€‰‘•™•¹Í¥Ù•}½Ù•É±…äˆ¤¤¤°(€€€€€€€Ñ¥µ•‘•±Ñ„ (€€€€€€€€€€€Í•½¹‘Ìõ¥¹Ð (€€€€€€€€€€€€€€€ÍÑÈ¡Ù…±Õ”¹•Ð ‰…‘•¹•}Í•½¹‘Ìˆ°Ù…±Õ•l‰‘•¥Í¥½¹}¥¹Ñ•ÉÙ…±}Í•½¹‘Ì‰t¤¤(€€€€€€€€€€€€¤(€€€€€€€€¤°(€€€€€€€Ñ¥µ•‘•±Ñ„¡Í•½¹‘Ìõ¥¹Ð¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰µ¥¹¥µÕµ}¡½±‘¥¹}Í•½¹‘Ìˆ°€À¤¤¤¤°(€€€€€€€Ñ¥µ•‘•±Ñ„¡Í•½¹‘Ìõ¥¹Ð¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰µ…á¥µÕµ}¡½±‘¥¹}Í•½¹‘Ìˆ°€À¤¤¤¤°(€€€€€€€Ñ¥µ•‘•±Ñ„¡Í•½¹‘Ìõ¥¹Ð¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰½½±‘½Ý¹}Í•½¹‘Ìˆ°€À¤¤¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰Ñ…É•Ñ}…±±½…Ñ¥½¹}Á•É•¹Ñ…”ˆ°€ˆÄÔˆ¤¤¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰•á¥Ñ}ÁÉ•‘¥Ñ½É}¹…µ”‰t¤(€€€€€€€¥˜Ù…±Õ”¹•Ð ‰•á¥Ñ}ÁÉ•‘¥Ñ½É}¹…µ”ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€½¹‘¥Ñ¥½¸¡ÍÑÈ¡Ù…±Õ•l‰•á¥Ñ}½¹‘¥Ñ¥½¸‰t¤¤(€€€€€€€¥˜Ù…±Õ”¹•Ð ‰•á¥Ñ}½¹‘¥Ñ¥½¸ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰•á¥Ñ}±½Ý•É}‰½Õ¹‰t¤¤(€€€€€€€¥˜Ù…±Õ”¹•Ð ‰•á¥Ñ}±½Ý•É}‰½Õ¹ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰•á¥Ñ}ÕÁÁ•É}‰½Õ¹‰t¤¤(€€€€€€€¥˜Ù…±Õ”¹•Ð ‰•á¥Ñ}ÕÁÁ•É}‰½Õ¹ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€ÑÕÁ±”¡ÍÑÈ¡¥Ñ•´¤™½È¥Ñ•´¥¸Í½ÕÉ•}¥‘Ì¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰•¹•É…Ñ¥½¹}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸ˆ°€Ä¤¤¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ”¹•Ð ‰ÅÕ…±¥™¥…Ñ¥½¹}½¹ÑÉ…Ñ}Ù•ÉÍ¥½¸ˆ°€Ä¤¤¤°(€€€€€€€½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}¹…µ”ô (€€€€€€€€€€€ÍÑÈ¡Ù…±Õ•l‰½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}¹…µ”‰t¤(€€€€€€€€€€€¥˜Ù…±Õ”¹•Ð ‰½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}¹…µ”ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€€€€½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸ô (€€€€€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸‰t¤¤(€€€€€€€€€€€¥˜Ù…±Õ”¹•Ð ‰½¹Ñ•áÑ}ÁÉ•‘¥Ñ½É}Ù•ÉÍ¥½¸ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€€€€½¹Ñ•áÑ}É•¥µ”ô (€€€€€€€€€€€ÍÑÈ¡Ù…±Õ•l‰½¹Ñ•áÑ}É•¥µ”‰t¤(€€€€€€€€€€€¥˜Ù…±Õ”¹•Ð ‰½¹Ñ•áÑ}É•¥µ”ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€€€€½¹Ñ•áÑ}±½Ý•É}‰½Õ¹ô (€€€€€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰½¹Ñ•áÑ}±½Ý•É}‰½Õ¹‰t¤¤(€€€€€€€€€€€¥˜Ù…±Õ”¹•Ð ‰½¹Ñ•áÑ}±½Ý•É}‰½Õ¹ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€€€€½¹Ñ•áÑ}ÕÁÁ•É}‰½Õ¹ô (€€€€€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰½¹Ñ•áÑ}ÕÁÁ•É}‰½Õ¹‰t¤¤(€€€€€€€€€€€¥˜Ù…±Õ”¹•Ð ‰½¹Ñ•áÑ}ÕÁÁ•É}‰½Õ¹ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€¤(()‘•˜}É•…‘}É…¹­¥¹œ¡Ù…±Õ”è½‰©•Ð¤€´ø‘•Õ¥‘•‘I…¹­¥¹œð9½¹”è(€€€¥˜¹½Ð¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‘¥Ð¤è(€€€€€€€É•ÑÕÉ¸9½¹”(€€€É•ÑÕÉ¸‘•Õ¥‘•‘I…¹­¥¹œ (€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰µ•‘¥…¹}Ù…±¥‘…Ñ¥½¹}É•ÑÕÉ¸‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰Ý½ÉÍÑ}Ù…±¥‘…Ñ¥½¹}É•ÑÕÉ¸‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰µ…á¥µÕµ}‘É…Ý‘½Ý¸‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰É•ÑÕÉ¹}½Ù•É}‘É…Ý‘½Ý¸‰t¤¤(€€€€€€€¥˜Ù…±Õ”¹•Ð ‰É•ÑÕÉ¹}½Ù•É}‘É…Ý‘½Ý¸ˆ¤¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰ÑÉ…‘•}½Õ¹Ð‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰Ñ½Ñ…±}™••Ì‰t¤¤°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰½ÍÑ}•™™¥¥•¹ä‰t¤¤(€€€€€€€¥˜Ù…±Õ”¹•Ð ‰½ÍÑ}•™™¥¥•¹äˆ¤¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”°(€€€€€€€•¥µ…°¡ÍÑÈ¡Ù…±Õ•l‰Á½Í¥Ñ¥Ù•}Ý¥¹‘½Ý}É…Ñ¥¼‰t¤¤°(€€€€€€€¥¹Ð¡ÍÑÈ¡Ù…±Õ•l‰Á½Í¥Ñ¥Ù•}Ý¥¹‘½Ý}½Õ¹Ð‰t¤¤°(€€€€€€€ÍÑÈ¡Ù…±Õ•l‰…¹‘¥‘…Ñ•}™¥¹•ÉÁÉ¥¹Ð‰t¤°(€€€€¤