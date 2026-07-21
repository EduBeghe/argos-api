
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.optimization.memory import StrategyCandidateRecord
from app.optimization.schemas import (
    CandidateDefinitionResponse,
    ScoreParametersResponse,
    SegmentBoundaryResponse,
    SegmentMetricsResponse,
    StrategyParametersResponse,
    WalkForwardBenchmarkResponse,
    WalkForwardCandidatePageResponse,
    WalkForwardConfigurationResponse,
    WalkForwardRunDetailResponse,
    WalkForwardRunResponse,
    WalkForwardSelectionPageResponse,
    WalkForwardSelectionResponse,
    WalkForwardStabilityResponse,
    WalkForwardWindowEvaluationPageResponse,
    WalkForwardWindowEvaluationResponse,
    WalkForwardWindowResponse,
)
from app.optimization.walk_forward import WalkForwardClassification
from app.optimization.walk_forward_memory import (
    WalkForwardBenchmarkEvaluationRecord,
    WalkForwardCandidateEvaluationRecord,
    WalkForwardCandidateSelectionRecord,
    WalkForwardCandidateStabilityRecord,
    WalkForwardRunRecord,
    WalkForwardWindowRecord,
)


class WalkForwardReadNotFoundError(LookupError):
    """Raised when requested walk-forward memory does not exist."""


async def read_refinement_walk_forward(
    session: AsyncSession, refinement_run_id: UUID
) -> WalkForwardRunResponse:
    run = await session.scalar(
        select(WalkForwardRunRecord)
        .where(WalkForwardRunRecord.refinement_run_id == refinement_run_id)
        .order_by(
            WalkForwardRunRecord.started_at.desc(), WalkForwardRunRecord.id.desc()
        )
        .limit(1)
    )
    if run is None:
        raise WalkForwardReadNotFoundError(
            f"walk-forward run for refinement run {refinement_run_id} was not found"
        )
    return _run_response(run)


async def read_walk_forward_run(
    session: AsyncSession, run_id: UUID
) -> WalkForwardRunDetailResponse:
    run = await session.get(WalkForwardRunRecord, run_id)
    if run is None:
        raise WalkForwardReadNotFoundError(f"walk-forward run {run_id} was not found")
    windows = (
        (
            await session.execute(
                select(WalkForwardWindowRecord)
                .where(WalkForwardWindowRecord.walk_forward_run_id == run_id)
                .order_by(WalkForwardWindowRecord.ordinal.asc())
                .limit(run.maximum_windows)
            )
        )
        .scalars()
        .all()
    )
    benchmarks = (
        (
            await session.execute(
                select(
                    WalkForwardBenchmarkEvaluationRecord,
                    WalkForwardWindowRecord.ordinal,
                )
                .join(
                    WalkForwardWindowRecord,
                    WalkForwardWindowRecord.id
                    == WalkForwardBenchmarkEvaluationRecord.window_id,
                )
                .where(
                    WalkForwardBenchmarkEvaluationRecord.walk_forward_run_id == run_id
                )
                .order_by(
                    WalkForwardWindowRecord.ordinal.asc(),
                    WalkForwardBenchmarkEvaluationRecord.benchmark.asc(),
                )
                .limit(run.maximum_windows * 2)
            )
        )
        .tuples()
        .all()
    )
    return WalkForwardRunDetailResponse(
        run=_run_response(run),
        windows=tuple(_window_response(window) for window in windows),
        benchmarks=tuple(
            _benchmark_response(benchmark, ordinal) for benchmark, ordinal in benchmarks
        ),
    )


async def read_walk_forward_candidates(
    session: AsyncSession,
    run_id: UUID,
    *,
    classification: WalkForwardClassification | None,
    limit: int,
    cursor: str | None,
) -> WalkForwardCandidatePageResponse:
    if await session.get(WalkForwardRunRecord, run_id) is None:
        raise WalkForwardReadNotFoundError(f"walk-forward run {run_id} was not found")
    statement = (
        select(WalkForwardCandidateStabilityRecord, StrategyCandidateRecord)
        .join(
            StrategyCandidateRecord,
            StrategyCandidateRecord.stable_candidate_id
            == WalkForwardCandidateStabilityRecord.stable_candidate_id,
        )
        .where(WalkForwardCandidateStabilityRecord.walk_forward_run_id == run_id)
    )
    if classification is not None:
        statement = statement.where(
            WalkForwardCandidateStabilityRecord.classification == classification.value
        )
    if cursor is not None:
        ranking, candidate_id = _decode_candidate_cursor(cursor)
        statement = statement.where(
            or_(
                WalkForwardCandidateStabilityRecord.ranking > ranking,
                and_(
                    WalkForwardCandidateStabilityRecord.ranking == ranking,
                    WalkForwardCandidateStabilityRecord.stable_candidate_id
                    > candidate_id,
                ),
            )
        )
    rows = (
        (
            await session.execute(
                statement.order_by(
                    WalkForwardCandidateStabilityRecord.ranking.asc(),
                    WalkForwardCandidateStabilityRecord.stable_candidate_id.asc(),

                ).limit(limit + 1)
            )
        )
        .tuples()
        .all()
    )
    page = rows[:limit]
    return WalkForwardCandidatePageResponse(
        items=tuple(
            _stability_response(stability, candidate) for stability, candidate in page
        ),
        has_more=len(rows) > limit,
        next_cursor=(
            f"{page[-1][0].ranking}:{page[-1][0].stable_candidate_id}"
            if len(rows) > limit and page
            else None
        ),
    )


async def read_walk_forward_candidate_windows(
    session: AsyncSession,
    run_id: UUID,
    candidate_id: str,
    *,
    limit: int,
    cursor: str | None,
) -> WalkForwardWindowEvaluationPageResponse:
    exists = await session.scalar(
        select(WalkForwardCandidateSelectionRecord.id).where(
            WalkForwardCandidateSelectionRecord.walk_forward_run_id == run_id,
            WalkForwardCandidateSelectionRecord.stable_candidate_id == candidate_id,
        )
    )
    if exists is None:
        raise WalkForwardReadNotFoundError(
            f"candidate {candidate_id} was not found in walk-forward run {run_id}"
        )
    statement = (
        select(
            WalkForwardCandidateSelectionRecord,
            WalkForwardCandidateEvaluationRecord,
            WalkForwardWindowRecord,
        )
        .join(
            WalkForwardWindowRecord,
            WalkForwardWindowRecord.id == WalkForwardCandidateSelectionRecord.window_id,
        )
        .outerjoin(
            WalkForwardCandidateEvaluationRecord,
            and_(
                WalkForwardCandidateEvaluationRecord.window_id
                == WalkForwardCandidateSelectionRecord.window_id,
                WalkForwardCandidateEvaluationRecord.stable_candidate_id
                == WalkForwardCandidateSelectionRecord.stable_candidate_id,
            ),
        )
        .where(
            WalkForwardCandidateSelectionRecord.walk_forward_run_id == run_id,
            WalkForwardCandidateSelectionRecord.stable_candidate_id == candidate_id,
        )
    )
    if cursor is not None:
        try:
            ordinal = int(cursor)
        except ValueError as error:
            raise ValueError("invalid walk-forward window cursor") from error
        statement = statement.where(WalkForwardWindowRecord.ordinal > ordinal)
    rows = (
        (
            await session.execute(
                statement.order_by(WalkForwardWindowRecord.ordinal.asc()).limit(
                    limit + 1
                )
            )
        )
        .tuples()
        .all()
    )
    page = rows[:limit]
    return WalkForwardWindowEvaluationPageResponse(
        items=tuple(
            _evaluation_response(selection, evaluation, window)
            for selection, evaluation, window in page
        ),
        has_more=len(rows) > limit,
        next_cursor=(str(page[-1][2].ordinal) if len(rows) > limit and page else None),
    )


async def read_walk_forward_selections(
    session: AsyncSession,
    run_id: UUID,
    *,
    selected_for_test: bool | None,
    limit: int,
    cursor: str | None,
) -> WalkForwardSelectionPageResponse:
    if await session.get(WalkForwardRunRecord, run_id) is None:
        raise WalkForwardReadNotFoundError(f"walk-forward run {run_id} was not found")
    statement = (
        select(
            WalkForwardCandidateSelectionRecord,
            WalkForwardWindowRecord,
            StrategyCandidateRecord,
        )
        .join(
            WalkForwardWindowRecord,
            WalkForwardWindowRecord.id == WalkForwardCandidateSelectionRecord.window_id,
        )
        .join(
            StrategyCandidateRecord,
            StrategyCandidateRecord.stable_candidate_id
            == WalkForwardCandidateSelectionRecord.stable_candidate_id,
        )
        .where(WalkForwardCandidateSelectionRecord.walk_forward_run_id == run_id)
    )
    if selected_for_test is not None:
        statement = statement.where(
            WalkForwardCandidateSelectionRecord.selected_for_test == selected_for_test
        )
    if cursor is not None:
        ordinal, rank, candidate_id = _decode_selection_cursor(cursor)
        statement = statement.where(
            or_(
                WalkForwardWindowRecord.ordinal > ordinal,
                and_(
                    WalkForwardWindowRecord.ordinal == ordinal,
                    WalkForwardCandidateSelectionRecord.validation_rank > rank,
                ),
                and_(
                    WalkForwardWindowRecord.ordinal == ordinal,
                    WalkForwardCandidateSelectionRecord.validation_rank == rank,
                    WalkForwardCandidateSelectionRecord.stable_candidate_id
                    > candidate_id,
                ),
            )
        )
    rows = (
        (
            await session.execute(
                statement.order_by(
                    WalkForwardWindowRecord.ordinal.asc(),
                    WalkForwardCandidateSelectionRecord.validation_rank.asc(),
                    WalkForwardCandidateSelectionRecord.stable_candidate_id.asc(),
                ).limit(limit + 1)
            )
        )
        .tuples()
        .all()

    )
    page = rows[:limit]
    return WalkForwardSelectionPageResponse(
        items=tuple(
            _selection_response(selection, window, candidate)
            for selection, window, candidate in page
        ),
        has_more=len(rows) > limit,
        next_cursor=(
            f"{page[-1][1].ordinal}:{page[-1][0].validation_rank}:"
            f"{page[-1][0].stable_candidate_id}"
            if len(rows) > limit and page
            else None
        ),
    )


def _run_response(run: WalkForwardRunRecord) -> WalkForwardRunResponse:
    return WalkForwardRunResponse(
        id=run.id,
        experiment_key=run.experiment_key,
        refinement_run_id=run.refinement_run_id,
        status=run.status,
        algorithm_version=run.algorithm_version,
        dataset_fingerprint=run.dataset_fingerprint,
        configuration=WalkForwardConfigurationResponse(
            **{
                name: getattr(run, name)
                for name in (
                    "training_days",
                    "validation_days",
                    "test_days",
                    "step_days",
                    "maximum_windows",
                    "maximum_candidates",
                    "minimum_valid_windows",
                    "minimum_profitable_window_ratio",
                    "maximum_drawdown",
                    "maximum_return_standard_deviation",
                )
            }
        ),
        initial_cash=run.initial_cash,
        fee_rate=run.fee_rate,
        slippage_rate=run.slippage_rate,
        decision_interval_seconds=run.decision_interval_seconds,
        windows_generated=run.windows_generated,
        windows_completed=run.windows_completed,
        candidates_evaluated=run.candidates_evaluated,
        evaluations_computed=run.evaluations_computed,
        evaluations_reused=run.evaluations_reused,
        compute_avoided_estimate=run.compute_avoided_estimate,
        recommended_candidate_id=run.recommended_candidate_id,
        recommendation_explanation=run.recommendation_explanation,
        started_at=run.started_at,
        completed_at=run.completed_at,
        elapsed_seconds=run.elapsed_seconds,
    )


def _window_response(window: WalkForwardWindowRecord) -> WalkForwardWindowResponse:
    return WalkForwardWindowResponse(
        id=window.id,
        ordinal=window.ordinal,
        training=SegmentBoundaryResponse(
            start=window.training_from, end=window.training_to
        ),
        validation=SegmentBoundaryResponse(
            start=window.validation_from, end=window.validation_to
        ),
        test=SegmentBoundaryResponse(start=window.test_from, end=window.test_to),
        decision_count=window.decision_count,
        dataset_fingerprint=window.dataset_fingerprint,
        selection_fingerprint=window.selection_fingerprint,
    )


def _definition(record: StrategyCandidateRecord) -> CandidateDefinitionResponse:
    return CandidateDefinitionResponse(
        stable_candidate_id=record.stable_candidate_id,
        score_model_name=record.score_model_name,
        score_model_version=record.score_model_version,
        strategy_name=record.strategy_name,
        strategy_version=record.strategy_version,
        score_parameters=ScoreParametersResponse.model_validate(
            record.score_parameters
        ),
        strategy_parameters=StrategyParametersResponse.model_validate(
            record.strategy_parameters
        ),
        created_at=record.created_at,
    )


def _stability_response(
    value: WalkForwardCandidateStabilityRecord, candidate: StrategyCandidateRecord
) -> WalkForwardStabilityResponse:
    return WalkForwardStabilityResponse(
        definition=_definition(candidate),
        **{
            name: getattr(value, name)
            for name in (
                "ranking",
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


def _evaluation_response(
    selection: WalkForwardCandidateSelectionRecord,
    evaluation: WalkForwardCandidateEvaluationRecord | None,
    window: WalkForwardWindowRecord,
) -> WalkForwardWindowEvaluationResponse:
    return WalkForwardWindowEvaluationResponse(
        id=evaluation.id if evaluation else None,
        window=_window_response(window),
        validation_rank=selection.validation_rank,
        selected_for_test=selection.selected_for_test,
        selection_reason=selection.selection_reason,
        validation_metrics=_validation_metrics(selection),
        status=evaluation.status if evaluation else "not_selected",
        total_return=evaluation.total_return if evaluation else None,
        absolute_pnl=evaluation.absolute_pnl if evaluation else None,
        max_drawdown=evaluation.max_drawdown if evaluation else None,
        final_equity=evaluation.final_equity if evaluation else None,
        fees_paid=evaluation.fees_paid if evaluation else None,
        trade_count=evaluation.trade_count if evaluation else None,
        return_over_drawdown=(evaluation.return_over_drawdown if evaluation else None),
        failure_reason=evaluation.failure_reason if evaluation else None,
    )


def _benchmark_response(

    value: WalkForwardBenchmarkEvaluationRecord, ordinal: int
) -> WalkForwardBenchmarkResponse:
    return WalkForwardBenchmarkResponse(
        id=value.id,
        window_ordinal=ordinal,
        benchmark=value.benchmark,
        status=value.status,
        total_return=value.total_return,
        absolute_pnl=value.absolute_pnl,
        max_drawdown=value.max_drawdown,
        final_equity=value.final_equity,
        fees_paid=value.fees_paid,
        trade_count=value.trade_count,
        return_over_drawdown=value.return_over_drawdown,
        failure_reason=value.failure_reason,
    )


def _decode_candidate_cursor(cursor: str) -> tuple[int, str]:
    try:
        ranking, candidate_id = cursor.split(":", maxsplit=1)
        if not candidate_id:
            raise ValueError
        return int(ranking), candidate_id
    except ValueError as error:
        raise ValueError("invalid walk-forward candidate cursor") from error


def _validation_metrics(
    value: WalkForwardCandidateSelectionRecord,
) -> SegmentMetricsResponse:
    return SegmentMetricsResponse(
        segment="validation",
        total_return=value.validation_total_return,
        absolute_pnl=value.validation_absolute_pnl,
        max_drawdown=value.validation_max_drawdown,
        trade_count=value.validation_trade_count,
        fees_paid=value.validation_fees_paid,
        final_equity=value.validation_final_equity,
        return_over_drawdown=value.validation_return_over_drawdown,
    )


def _selection_response(
    selection: WalkForwardCandidateSelectionRecord,
    window: WalkForwardWindowRecord,
    candidate: StrategyCandidateRecord,
) -> WalkForwardSelectionResponse:
    return WalkForwardSelectionResponse(
        definition=_definition(candidate),
        window=_window_response(window),
        validation_rank=selection.validation_rank,
        selected_for_test=selection.selected_for_test,
        selection_reason=selection.selection_reason,
        validation_metrics=_validation_metrics(selection),
    )


def _decode_selection_cursor(cursor: str) -> tuple[int, int, str]:
    try:
        ordinal, rank, candidate_id = cursor.split(":", maxsplit=2)
        if not candidate_id:
            raise ValueError
        return int(ordinal), int(rank), candidate_id
    except ValueError as error:
        raise ValueError("invalid walk-forward selection cursor") from error
