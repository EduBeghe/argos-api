from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.experience import (
    CandidateHistoricalProfile,
    HistoricalEvaluation,
    build_candidate_profiles,
)
from app.optimization.memory import (
    StrategyBenchmarkEvaluationRecord,
    StrategyCandidateEvaluationRecord,
    StrategyCandidateRecord,
    StrategyRefinementRunRecord,
)
from app.optimization.queries import (
    CANDIDATE_SCAN_BATCH_SIZE,
    get_candidate,
    get_refinement_run,
    list_candidate_definition_batch,
    list_refinement_runs,
    load_candidate_history,
    load_profile_evidence,
    load_run_benchmarks,
    load_run_evaluation_counts,
    load_run_evaluation_page,
)
from app.optimization.schemas import (
    CandidateDefinitionResponse,
    CandidateDetailResponse,
    CandidateEvaluationHistoryResponse,
    CandidateEvaluationPageResponse,
    CandidatePageResponse,
    CandidateProfileResponse,
    CandidateQuery,
    CandidateSummaryResponse,
    EvaluationGroupCountResponse,
    EvaluationPageResponse,
    EvaluationResponse,
    InstrumentIdentityResponse,
    PlannerAllocationResponse,
    RefinementBudgetResponse,
    RefinementCountsResponse,
    RefinementRunDetailResponse,
    RefinementRunQuery,
    RefinementRunSummaryResponse,
    ScoreParametersResponse,
    SearchPlanResponse,
    SegmentBoundaryResponse,
    SegmentMetricsResponse,
    StrategyParametersResponse,
    TimeRangeResponse,
)

PUBLIC_FAILURE_REASON = (
    "Experiment failed; operational details are available in server logs."
)


class OptimizationReadNotFoundError(LookupError):
    """Raised when requested persisted optimization evidence does not exist."""


async def read_refinement_runs(
    session: AsyncSession, query: RefinementRunQuery
) -> tuple[RefinementRunSummaryResponse, ...]:
    return tuple(
        _run_summary(run) for run in await list_refinement_runs(session, query)
    )


async def read_refinement_run(
    session: AsyncSession, run_id: UUID
) -> RefinementRunDetailResponse:
    run = await get_refinement_run(session, run_id)
    if run is None:
        raise OptimizationReadNotFoundError("refinement run not found")
    evaluation_counts = await load_run_evaluation_counts(session, run_id)
    benchmarks = await load_run_benchmarks(session, run_id)
    recommended = (
        await get_candidate(session, run.recommended_candidate_id)
        if run.recommended_candidate_id is not None
        else None
    )
    return RefinementRunDetailResponse(
        run=_run_summary(run),
        search_plan=_search_plan(run.search_plan),
        production_baseline=tuple(
            _metrics(value)
            for value in benchmarks
            if value.benchmark == "production_baseline"
        ),
        buy_and_hold=tuple(
            _metrics(value) for value in benchmarks if value.benchmark == "buy_and_hold"
        ),
        recommended_candidate=(
            _candidate_definition(recommended) if recommended is not None else None
        ),
        explanation=run.recommendation_explanation,
        result_digest=_result_digest(run.result_payload),
        evaluation_summary=tuple(
            EvaluationGroupCountResponse(phase=phase, segment=segment, count=count)
            for phase, segment, count in evaluation_counts
        ),
    )


async def read_run_evaluations(
    session: AsyncSession,
    run_id: UUID,
    *,
    limit: int,
    cursor: str | None,
) -> EvaluationPageResponse:
    if await get_refinement_run(session, run_id) is None:
        raise OptimizationReadNotFoundError("refinement run not found")
    rows = await load_run_evaluation_page(
        session, run_id, after_id=_uuid_cursor(cursor), limit=limit
    )
    has_more = len(rows) > limit
    page = rows[:limit]
    return EvaluationPageResponse(
        items=tuple(_evaluation(item) for item in page),
        has_more=has_more,
        next_cursor=str(page[-1].id) if has_more and page else None,
    )


async def read_candidates(
    session: AsyncSession, query: CandidateQuery
) -> CandidatePageResponse:
    selected: list[CandidateSummaryResponse] = []
    scan_cursor = query.cursor
    while len(selected) <= query.limit:
        definitions = await list_candidate_definition_batch(
            session,
            query,
            after_candidate_id=scan_cursor,
            batch_size=CANDIDATE_SCAN_BATCH_SIZE,
        )
        if not definitions:
            break
        rows = await load_profile_evidence(
            session,
            candidate_ids=tuple(value.stable_candidate_id for value in definitions),
            query=query,
        )
        profiles = _profiles(rows)
        for definition in definitions:
            profile = profiles.get(definition.stable_candidate_id)
            if profile is not None and _profile_matches(profile, query):
                selected.append(
                    CandidateSummaryResponse(
                        definition=_candidate_definition(definition),
                        profile=_profile(profile),
                    )
                )
                if len(selected) > query.limit:
                    break
        if len(selected) > query.limit:
            break
        scan_cursor = definitions[-1].stable_candidate_id
        if len(definitions) < CANDIDATE_SCAN_BATCH_SIZE:
            break
    has_more = len(selected) > query.limit
    page = selected[: query.limit]
    return CandidatePageResponse(
        items=tuple(page),
        has_more=has_more,
        next_cursor=(
            page[-1].definition.stable_candidate_id if has_more and page else None
        ),
    )


async def read_candidate(
    session: AsyncSession,
    candidate_id: str,
    *,
    limit: int,
    cursor: str | None,
) -> CandidateDetailResponse:
    definition = await get_candidate(session, candidate_id)
    if definition is None:
        raise OptimizationReadNotFoundError("optimization candidate not found")
    profile_rows = await load_profile_evidence(session, candidate_ids=(candidate_id,))
    profiles = _profiles(profile_rows)
    if candidate_id not in profiles:
        raise OptimizationReadNotFoundError("candidate has no completed evidence")
    history = await load_candidate_history(
        session, candidate_id, limit=limit, after=_history_cursor(cursor)
    )
    run_ids = tuple(sorted({run.id for _, run, _ in profile_rows}, key=str))
    return CandidateDetailResponse(
        definition=_candidate_definition(definition),
        profile=_profile(profiles[candidate_id]),
        was_ever_recommended=any(
            run.recommended_candidate_id == candidate_id for _, run, _ in profile_rows
        ),
        related_run_ids=run_ids,
        evaluations=CandidateEvaluationPageResponse(
            items=tuple(
                CandidateEvaluationHistoryResponse(
                    **_evaluation(evaluation).model_dump(),
                    run_id=run.id,
                    instrument=_instrument(run),
                    excess_return_against_baseline=evaluation.total_return - baseline,
                    was_recommended=run.recommended_candidate_id == candidate_id,
                )
                for evaluation, run, baseline in history[:limit]
            ),
            has_more=len(history) > limit,
            next_cursor=(
                _encode_history_cursor(history[limit - 1][0])
                if len(history) > limit
                else None
            ),
        ),
    )


def _profile_matches(
    profile: CandidateHistoricalProfile, query: CandidateQuery
) -> bool:
    return profile.validation_evaluation_count >= query.minimum_validation_runs and (
        query.evidence_classification is None
        or profile.evidence_classification is query.evidence_classification
    )


def _uuid_cursor(value: str | None) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(value)
    except ValueError as error:
        raise ValueError("invalid evaluation cursor") from error


def _history_cursor(value: str | None) -> tuple[datetime, UUID] | None:
    if value is None:
        return None
    try:
        timestamp, identifier = value.rsplit("|", 1)
        return datetime.fromisoformat(timestamp), UUID(identifier)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid candidate evaluation cursor") from error


def _encode_history_cursor(value: StrategyCandidateEvaluationRecord) -> str:
    return f"{value.evaluated_at.isoformat()}|{value.id}"


def _profiles(
    rows: list[
        tuple[StrategyCandidateEvaluationRecord, StrategyRefinementRunRecord, Decimal]
    ],
) -> dict[str, CandidateHistoricalProfile]:
    evaluations = tuple(
        HistoricalEvaluation(
            evaluation.stable_candidate_id,
            str(run.id),
            Instrument(
                Asset(run.asset),
                Provider(run.provider),
                MarketType(run.market_type),
                Interval(run.interval),
            ),
            evaluation.segment,
            evaluation.total_return,
            baseline,
            evaluation.max_drawdown,
            evaluation.trade_count,
            evaluation.fees_paid,
            evaluation.eligible,
            evaluation.evaluated_at,
            evaluation.eligible if evaluation.segment == "test" else None,
        )
        for evaluation, run, baseline in rows
    )
    return {
        value.stable_candidate_id: value
        for value in build_candidate_profiles(evaluations)
    }


def _run_summary(run: StrategyRefinementRunRecord) -> RefinementRunSummaryResponse:
    return RefinementRunSummaryResponse(
        id=run.id,
        experiment_key=run.experiment_key,
        status=run.status,
        instrument=_instrument(run),
        requested_range=TimeRangeResponse(
            start=run.requested_from, end=run.requested_to
        ),
        training=SegmentBoundaryResponse(start=run.training_from, end=run.training_to),
        validation=SegmentBoundaryResponse(
            start=run.validation_from, end=run.validation_to
        ),
        test=SegmentBoundaryResponse(start=run.test_from, end=run.test_to),
        score_model_name=run.score_model_name,
        score_model_version=run.score_model_version,
        strategy_name=run.strategy_name,
        strategy_version=run.strategy_version,
        simulation_engine_version=run.simulation_engine_version,
        planner_version=run.search_planner_version,
        initial_cash=run.initial_cash,
        account_currency=run.account_currency,
        fee_rate=run.fee_rate,
        slippage_rate=run.slippage_rate,
        decision_interval_seconds=run.decision_interval_seconds,
        refinement_budget=RefinementBudgetResponse.model_validate(
            run.refinement_budget
        ),
        planner_allocation=PlannerAllocationResponse.model_validate(
            run.planner_allocation
        ),
        history_cutoff=run.history_cutoff,
        recommendation_status=run.recommendation_status,
        recommended_candidate_id=run.recommended_candidate_id,
        counts=(
            RefinementCountsResponse.model_validate(run.counts) if run.counts else None
        ),
        elapsed_seconds=run.elapsed_seconds,
        started_at=run.started_at,
        completed_at=run.completed_at,
        failed_at=run.failed_at,
        failure_reason=PUBLIC_FAILURE_REASON if run.failure_reason else None,
        last_verified_at=run.last_verified_at,
    )


def _instrument(run: StrategyRefinementRunRecord) -> InstrumentIdentityResponse:
    return InstrumentIdentityResponse(
        asset=Asset(run.asset),
        provider=Provider(run.provider),
        market_type=MarketType(run.market_type),
        interval=Interval(run.interval),
    )


def _candidate_definition(
    value: StrategyCandidateRecord,
) -> CandidateDefinitionResponse:
    return CandidateDefinitionResponse(
        stable_candidate_id=value.stable_candidate_id,
        score_model_name=value.score_model_name,
        score_model_version=value.score_model_version,
        strategy_name=value.strategy_name,
        strategy_version=value.strategy_version,
        score_parameters=ScoreParametersResponse.model_validate(value.score_parameters),
        strategy_parameters=StrategyParametersResponse.model_validate(
            value.strategy_parameters
        ),
        created_at=value.created_at,
    )


def _profile(value: CandidateHistoricalProfile) -> CandidateProfileResponse:
    return CandidateProfileResponse(
        evidence_classification=value.evidence_classification,
        profile_version=value.profile_version,
        compatible_run_count=value.compatible_run_count,
        validation_evaluation_count=value.validation_evaluation_count,
        test_evaluation_count=value.test_evaluation_count,
        validation_win_rate=value.validation_win_rate,
        test_confirmation_rate=value.test_confirmation_rate,
        median_validation_return=value.median_validation_return,
        median_test_return=value.median_test_return,
        median_validation_excess_return=value.median_validation_excess_return,
        median_test_excess_return=value.median_test_excess_return,
        worst_validation_drawdown=value.worst_validation_drawdown,
        worst_test_drawdown=value.worst_test_drawdown,
        median_trade_count=value.median_trade_count,
        median_fees_paid=value.median_fees_paid,
        first_evaluated_at=value.first_evaluated_at,
        last_evaluated_at=value.last_evaluated_at,
        consecutive_recent_failures=value.consecutive_recent_failures,
        instruments=tuple(
            InstrumentIdentityResponse.from_instrument(item)
            for item in value.instruments
        ),
    )


def _metrics(
    value: StrategyBenchmarkEvaluationRecord,
) -> SegmentMetricsResponse:
    return SegmentMetricsResponse(
        segment=value.segment,
        total_return=value.total_return,
        absolute_pnl=value.absolute_pnl,
        max_drawdown=value.max_drawdown,
        trade_count=value.trade_count,
        fees_paid=value.fees_paid,
        final_equity=value.final_equity,
        return_over_drawdown=value.return_over_drawdown,
    )


def _evaluation(value: StrategyCandidateEvaluationRecord) -> EvaluationResponse:
    return EvaluationResponse(
        id=value.id,
        stable_candidate_id=value.stable_candidate_id,
        phase=value.phase,
        segment=value.segment,
        disabled_signal=value.disabled_signal,
        total_return=value.total_return,
        absolute_pnl=value.absolute_pnl,
        max_drawdown=value.max_drawdown,
        trade_count=value.trade_count,
        fees_paid=value.fees_paid,
        final_equity=value.final_equity,
        return_over_drawdown=value.return_over_drawdown,
        rank=value.rank,
        eligible=value.eligible,
        rejection_reason=value.rejection_reason,
        evaluated_at=value.evaluated_at,
    )


def _search_plan(value: dict[str, object] | None) -> SearchPlanResponse | None:
    if not value:
        return None
    categories = {
        name: _string_tuple(value.get(name))
        for name in (
            "known_rechecks",
            "promising_neighborhoods",
            "uncertain_rechecks",
            "novel_exploration",
        )
    }
    return SearchPlanResponse(
        planner_version=_integer(value.get("planner_version")),
        **categories,
        known_recheck_count=len(categories["known_rechecks"]),
        promising_neighborhood_count=len(categories["promising_neighborhoods"]),
        uncertain_recheck_count=len(categories["uncertain_rechecks"]),
        novel_exploration_count=len(categories["novel_exploration"]),
    )


def _result_digest(value: dict[str, object] | None) -> str | None:
    if not value:
        return None
    digest = value.get("result_digest")
    return str(digest) if digest is not None else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
