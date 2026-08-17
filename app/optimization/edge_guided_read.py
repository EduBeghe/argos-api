from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import Subquery

from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization._persistence_helpers import require
from app.optimization.edge_guided import (
    EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION,
    CandidateFamily,
    EdgeGuidedCandidateStatus,
    EdgeGuidedRejectionReason,
    EntryRule,
    ExitRule,
    ResearchOutcome,
)
from app.optimization.edge_guided_memory import (
    EdgeGuidedCandidateRecord,
    EdgeGuidedRunRecord,
    EdgeGuidedWindowRecord,
)
from app.optimization.edge_guided_schemas import (
    EdgeGuidedActionFilter,
    EdgeGuidedCandidateDetailResponse,
    EdgeGuidedCandidatePageResponse,
    EdgeGuidedCandidateResponse,
    EdgeGuidedRankingResponse,
    EdgeGuidedRunPageResponse,
    EdgeGuidedRunResponse,
    EdgeGuidedSegmentMetricsResponse,
    EdgeGuidedStatusCountsResponse,
    EdgeGuidedWindowResponse,
)
from app.optimization.predictive_edges import (
    Condition,
    DirectionMapping,
    SuggestedAction,
)
from app.optimization.refinement import RefinementSegmentName
from app.optimization.schemas import InstrumentIdentityResponse

DEFAULT_EDGE_GUIDED_PAGE_LIMIT = 50
MAX_EDGE_GUIDED_PAGE_LIMIT = 100


class EdgeGuidedReadNotFoundError(LookupError):
    """Raised when persisted edge-guided evidence does not exist."""


def _status_counts_subquery() -> Subquery:
    return (
        select(
            EdgeGuidedCandidateRecord.run_id.label("run_id"),
            func.count()
            .filter(
                EdgeGuidedCandidateRecord.status
                == EdgeGuidedCandidateStatus.CONFIRMED.value
            )
            .label("confirmed_candidate"),
            func.count()
            .filter(
                EdgeGuidedCandidateRecord.status
                == EdgeGuidedCandidateStatus.REJECTED_VALIDATION.value
            )
            .label("rejected_validation"),
            func.count()
            .filter(
                EdgeGuidedCandidateRecord.status
                == EdgeGuidedCandidateStatus.REJECTED_TEST.value
            )
            .label("rejected_test"),
        )
        .group_by(EdgeGuidedCandidateRecord.run_id)
        .subquery()
    )


async def list_edge_guided_runs(
    session: AsyncSession,
    *,
    asset: Asset | None = None,
    provider: Provider | None = None,
    market_type: MarketType | None = None,
    interval: Interval | None = None,
    status: str | None = None,
    limit: int = DEFAULT_EDGE_GUIDED_PAGE_LIMIT,
    offset: int = 0,
) -> EdgeGuidedRunPageResponse:
    counts = _status_counts_subquery()
    statement = select(
        EdgeGuidedRunRecord,
        counts.c.confirmed_candidate,
        counts.c.rejected_validation,
        counts.c.rejected_test,
    ).outerjoin(counts, counts.c.run_id == EdgeGuidedRunRecord.id)
    conditions = _run_conditions(asset, provider, market_type, interval, status)
    statement = statement.where(*conditions)
    total = await session.scalar(
        select(func.count()).select_from(EdgeGuidedRunRecord).where(*conditions)
    )
    rows = (
        await session.execute(
            statement.order_by(
                EdgeGuidedRunRecord.completed_at.desc().nullslast(),
                EdgeGuidedRunRecord.started_at.desc(),
                EdgeGuidedRunRecord.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return EdgeGuidedRunPageResponse(
        items=tuple(
            _run(row[0], row[1] or 0, row[2] or 0, row[3] or 0) for row in rows
        ),
        total=total or 0,
        limit=limit,
        offset=offset,
    )


async def get_edge_guided_run(
    session: AsyncSession, run_id: UUID
) -> EdgeGuidedRunResponse:
    counts = _status_counts_subquery()
    row = (
        await session.execute(
            select(
                EdgeGuidedRunRecord,
                counts.c.confirmed_candidate,
                counts.c.rejected_validation,
                counts.c.rejected_test,
            )
            .outerjoin(counts, counts.c.run_id == EdgeGuidedRunRecord.id)
            .where(EdgeGuidedRunRecord.id == run_id)
        )
    ).first()
    row = require(row, EdgeGuidedReadNotFoundError("edge-guided run was not found"))
    return _run(row[0], row[1] or 0, row[2] or 0, row[3] or 0)


async def list_edge_guided_candidates(
    session: AsyncSession,
    *,
    run_id: UUID | None = None,
    asset: Asset | None = None,
    provider: Provider | None = None,
    market_type: MarketType | None = None,
    interval: Interval | None = None,
    status: EdgeGuidedCandidateStatus | None = None,
    action: EdgeGuidedActionFilter | None = None,
    predictor_name: str | None = None,
    condition: Condition | None = None,
    limit: int = DEFAULT_EDGE_GUIDED_PAGE_LIMIT,
    offset: int = 0,
) -> EdgeGuidedCandidatePageResponse:
    statement = select(EdgeGuidedCandidateRecord, EdgeGuidedRunRecord).join(
        EdgeGuidedRunRecord, EdgeGuidedRunRecord.id == EdgeGuidedCandidateRecord.run_id
    )
    conditions = _candidate_conditions(
        run_id,
        asset,
        provider,
        market_type,
        interval,
        status,
        action,
        predictor_name,
        condition,
    )
    statement = statement.where(*conditions)
    total_statement = (
        select(func.count())
        .select_from(EdgeGuidedCandidateRecord)
        .join(
            EdgeGuidedRunRecord,
            EdgeGuidedRunRecord.id == EdgeGuidedCandidateRecord.run_id,
        )
        .where(*conditions)
    )
    total = await session.scalar(total_statement)
    rows = (
        await session.execute(
            statement.order_by(
                EdgeGuidedRunRecord.completed_at.desc().nullslast(),
                EdgeGuidedCandidateRecord.run_id.desc(),
                EdgeGuidedCandidateRecord.candidate_fingerprint.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return EdgeGuidedCandidatePageResponse(
        items=tuple(_candidate(candidate, run) for candidate, run in rows),
        total=total or 0,
        limit=limit,
        offset=offset,
    )


async def get_edge_guided_candidate(
    session: AsyncSession, candidate_evaluation_id: UUID
) -> EdgeGuidedCandidateDetailResponse:
    row = (
        await session.execute(
            select(EdgeGuidedCandidateRecord, EdgeGuidedRunRecord)
            .join(
                EdgeGuidedRunRecord,
                EdgeGuidedRunRecord.id == EdgeGuidedCandidateRecord.run_id,
            )
            .where(EdgeGuidedCandidateRecord.id == candidate_evaluation_id)
        )
    ).first()
    row = require(
        row, EdgeGuidedReadNotFoundError("edge-guided candidate was not found")
    )
    candidate, run = row
    windows = (
        await session.scalars(
            select(EdgeGuidedWindowRecord)
            .where(
                EdgeGuidedWindowRecord.candidate_evaluation_id
                == candidate_evaluation_id
            )
            .order_by(EdgeGuidedWindowRecord.window_ordinal.asc())
        )
    ).all()
    return EdgeGuidedCandidateDetailResponse(
        **_candidate(candidate, run).model_dump(),
        windows=tuple(_window(item) for item in windows),
    )


def _run_conditions(
    asset: Asset | None,
    provider: Provider | None,
    market_type: MarketType | None,
    interval: Interval | None,
    status: str | None,
) -> tuple[ColumnElement[bool], ...]:
    conditions: list[ColumnElement[bool]] = []
    for column, value in (
        (EdgeGuidedRunRecord.asset, asset.value if asset else None),
        (EdgeGuidedRunRecord.provider, provider.value if provider else None),
        (EdgeGuidedRunRecord.market_type, market_type.value if market_type else None),
        (EdgeGuidedRunRecord.interval, interval.value if interval else None),
        (EdgeGuidedRunRecord.status, status),
    ):
        if value is not None:
            conditions.append(column == value)
    return tuple(conditions)


def _candidate_conditions(
    run_id: UUID | None,
    asset: Asset | None,
    provider: Provider | None,
    market_type: MarketType | None,
    interval: Interval | None,
    status: EdgeGuidedCandidateStatus | None,
    action: EdgeGuidedActionFilter | None,
    predictor_name: str | None,
    condition: Condition | None,
) -> tuple[ColumnElement[bool], ...]:
    conditions: list[ColumnElement[bool]] = []
    for column, value in (
        (EdgeGuidedCandidateRecord.run_id, run_id),
        (EdgeGuidedRunRecord.asset, asset.value if asset else None),
        (EdgeGuidedRunRecord.provider, provider.value if provider else None),
        (EdgeGuidedRunRecord.market_type, market_type.value if market_type else None),
        (EdgeGuidedRunRecord.interval, interval.value if interval else None),
        (EdgeGuidedCandidateRecord.status, status.value if status else None),
        (EdgeGuidedCandidateRecord.suggested_action, action.value if action else None),
        (EdgeGuidedCandidateRecord.predictor_name, predictor_name),
        (EdgeGuidedCandidateRecord.condition, condition.value if condition else None),
    ):
        if value is not None:
            conditions.append(column == value)
    return tuple(conditions)


def _run(
    record: EdgeGuidedRunRecord,
    confirmed_candidate: int,
    rejected_validation: int,
    rejected_test: int,
) -> EdgeGuidedRunResponse:
    return EdgeGuidedRunResponse(
        id=record.id,
        experiment_key=record.experiment_key,
        predictive_edge_run_id=record.predictive_edge_run_id,
        status=record.status,
        instrument=InstrumentIdentityResponse(
            asset=Asset(record.asset),
            provider=Provider(record.provider),
            market_type=MarketType(record.market_type),
            interval=Interval(record.interval),
        ),
        dataset_fingerprint=record.dataset_fingerprint,
        range_start=record.range_start,
        range_end=record.range_end,
        initial_cash=record.initial_cash,
        account_currency=record.account_currency,
        fee_rate=record.fee_rate,
        slippage_rate=record.slippage_rate,
        decision_interval_seconds=record.decision_interval_seconds,
        maximum_candidates=record.maximum_candidates,
        algorithm_version=record.algorithm_version,
        qualification_contract_version=record.qualification_contract_version,
        legacy=(
            record.qualification_contract_version
            < EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
        ),
        generated_count=record.generated_count,
        evaluated_count=record.evaluated_count,
        windows_available=record.windows_available,
        started_at=record.started_at,
        completed_at=record.completed_at,
        candidate_status_counts=EdgeGuidedStatusCountsResponse(
            confirmed_candidate=confirmed_candidate,
            rejected_validation=rejected_validation,
            rejected_test=rejected_test,
        ),
        research_outcome=(
            ResearchOutcome(record.research_outcome)
            if record.research_outcome
            else None
        ),
        outcome_explanation=record.outcome_explanation,
        funnel={key: int(str(value)) for key, value in (record.funnel or {}).items()}
        if record.funnel is not None
        else None,
        timings={
            key: Decimal(str(value)) for key, value in (record.timings or {}).items()
        }
        if record.timings is not None
        else None,
    )


def _metric(value: dict[str, object]) -> EdgeGuidedSegmentMetricsResponse:
    total_return = Decimal(str(value["total_return"]))
    max_drawdown = Decimal(str(value["max_drawdown"]))
    trade_count = int(str(value["trade_count"]))
    production_difference = Decimal(str(value["return_difference_from_production"]))
    buy_and_hold_difference = Decimal(str(value["return_difference_from_buy_and_hold"]))
    return EdgeGuidedSegmentMetricsResponse(
        segment=RefinementSegmentName(str(value["segment"])),
        total_return=total_return,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        fees_paid=Decimal(str(value["fees_paid"])),
        return_difference_from_production=production_difference,
        return_difference_from_buy_and_hold=buy_and_hold_difference,
        sufficient_trades=bool(value.get("sufficient_trades", trade_count > 0)),
        positive_net_return=bool(value.get("positive_net_return", total_return > 0)),
        beats_production=bool(value.get("beats_production", production_difference > 0)),
        beats_buy_and_hold=bool(
            value.get("beats_buy_and_hold", buy_and_hold_difference > 0)
        ),
        risk_acceptable=bool(
            value.get("risk_acceptable", max_drawdown <= Decimal("0.10"))
        ),
    )


def _candidate(
    record: EdgeGuidedCandidateRecord, run: EdgeGuidedRunRecord
) -> EdgeGuidedCandidateResponse:
    payload = record.candidate
    ranking_payload = payload.get("ranking")
    ranking = (
        EdgeGuidedRankingResponse.model_validate(ranking_payload)
        if isinstance(ranking_payload, dict)
        else None
    )
    return EdgeGuidedCandidateResponse(
        candidate_evaluation_id=record.id,
        run_id=record.run_id,
        candidate_fingerprint=record.candidate_fingerprint,
        source_hypothesis_id=record.source_hypothesis_id,
        source_edge_fingerprint=str(payload["source_edge_fingerprint"]),
        asset=run.asset,
        provider=run.provider,
        market_type=run.market_type,
        interval=run.interval,
        dataset_fingerprint=run.dataset_fingerprint,
        range_start=run.range_start,
        range_end=run.range_end,
        predictor_name=record.predictor_name,
        predictor_version=record.predictor_version,
        hypothesis_contract_version=int(str(payload["hypothesis_contract_version"])),
        condition=Condition(record.condition),
        lower_bound=record.bin_lower_bound,
        upper_bound=record.bin_upper_bound,
        suggested_action=SuggestedAction(record.suggested_action),
        horizon_seconds=record.horizon_seconds,
        decision_interval_seconds=int(str(payload["decision_interval_seconds"])),
        direction_mapping=DirectionMapping(record.direction_mapping),
        score_model_name=record.score_model_name,
        score_model_version=record.score_model_version,
        strategy_name=record.strategy_name,
        strategy_version=record.strategy_version,
        strategy_configuration_version=record.strategy_configuration_version,
        account_currency=str(payload["account_currency"]),
        fee_rate=Decimal(str(payload["fee_rate"])),
        slippage_rate=Decimal(str(payload["slippage_rate"])),
        status=EdgeGuidedCandidateStatus(record.status),
        qualification_contract_version=run.qualification_contract_version,
        legacy=(
            run.qualification_contract_version
            < EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
        ),
        eligible_under_current_contract=(
            run.qualification_contract_version
            == EDGE_GUIDED_QUALIFICATION_CONTRACT_VERSION
            and record.status == EdgeGuidedCandidateStatus.CONFIRMED.value
        ),
        rejection_reasons=tuple(
            EdgeGuidedRejectionReason(value) for value in record.rejection_reasons
        ),
        explanation=record.explanation,
        unconditional_benchmark=record.unconditional_benchmark,
        valid_window_count=record.valid_window_count,
        confirmed_window_count=record.confirmed_window_count,
        rejected_validation_window_count=record.rejected_validation_window_count,
        rejected_test_window_count=record.rejected_test_window_count,
        training_metrics=_metric(record.training_metrics),
        validation_metrics=_metric(record.validation_metrics),
        test_metrics=_metric(record.test_metrics) if record.test_metrics else None,
        candidate_family=CandidateFamily(str(payload["candidate_family"]))
        if payload.get("candidate_family") is not None
        else None,
        entry_rule=EntryRule(str(payload["entry_rule"]))
        if payload.get("entry_rule") is not None
        else None,
        exit_rule=ExitRule(str(payload["exit_rule"]))
        if payload.get("exit_rule") is not None
        else None,
        cadence_seconds=int(str(payload["cadence_seconds"]))
        if payload.get("cadence_seconds") is not None
        else None,
        minimum_holding_seconds=int(str(payload["minimum_holding_seconds"]))
        if payload.get("minimum_holding_seconds") is not None
        else None,
        maximum_holding_seconds=int(str(payload["maximum_holding_seconds"]))
        if payload.get("maximum_holding_seconds") is not None
        else None,
        cooldown_seconds=int(str(payload["cooldown_seconds"]))
        if payload.get("cooldown_seconds") is not None
        else None,
        target_allocation_percentage=Decimal(
            str(payload["target_allocation_percentage"])
        )
        if payload.get("target_allocation_percentage") is not None
        else None,
        generation_contract_version=int(str(payload["generation_contract_version"]))
        if payload.get("generation_contract_version") is not None
        else None,
        ranking=ranking,
        cost_efficiency=ranking.cost_efficiency if ranking else None,
        positive_window_ratio=ranking.positive_window_ratio if ranking else None,
        worst_window_return=ranking.worst_validation_return if ranking else None,
    )


def _window(record: EdgeGuidedWindowRecord) -> EdgeGuidedWindowResponse:
    return EdgeGuidedWindowResponse(
        window_ordinal=record.window_ordinal,
        volatility_threshold=record.volatility_threshold,
        training_metrics=_metric(record.training_metrics),
        validation_metrics=_metric(record.validation_metrics),
        test_metrics=_metric(record.test_metrics) if record.test_metrics else None,
        validation_passed=record.validation_passed,
        test_confirmed=record.test_confirmed,
        rejection_reasons=tuple(
            EdgeGuidedRejectionReason(value) for value in record.rejection_reasons
        ),
    )
