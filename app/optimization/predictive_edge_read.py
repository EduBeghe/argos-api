from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.optimization.predictive_edge_memory import (
    PredictiveEdgeAggregateRecord,
    PredictiveEdgeBinAggregateRecord,
    PredictiveEdgeNearMissRecord,
    PredictiveEdgeRunDiagnosticRecord,
    PredictiveEdgeRunRecord,
    PredictiveEdgeSensitivityRecord,
    PredictiveEdgeWindowMetricRecord,
    PredictiveEdgeWindowRecord,
    StrategyHypothesisRecord,
)
from app.optimization.predictive_edge_schemas import (
    PredictiveEdgeBinSelectionResponse,
    PredictiveEdgeDiagnosticResponse,
    PredictiveEdgeNearMissDetailResponse,
    PredictiveEdgeNearMissPageResponse,
    PredictiveEdgeNearMissResponse,
    PredictiveEdgePageResponse,
    PredictiveEdgeResponse,
    PredictiveEdgeRunDetailResponse,
    PredictiveEdgeRunPageResponse,
    PredictiveEdgeRunResponse,
    PredictiveEdgeWindowMetricResponse,
    PredictiveEdgeWindowPageResponse,
    StrategyHypothesisPageResponse,
    StrategyHypothesisResponse,
    ThresholdSensitivityResponse,
)
from app.optimization.predictive_edges import (
    DirectionMapping,
    EvidenceStatus,
    PredictorSemanticKind,
    SuggestedAction,
    TestConfirmationStatus,
    ValidationStatus,
)

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


class PredictiveEdgeReadError(ValueError):
    """Raised for invalid predictive-edge cursors or limits."""


class PredictiveEdgeNotFoundError(LookupError):
    """Raised when predictive-edge evidence does not exist."""


def _limit(value: int) -> int:
    if not 1 <= value <= MAX_PAGE_LIMIT:
        raise PredictiveEdgeReadError(f"limit must be between 1 and {MAX_PAGE_LIMIT}")
    return value


async def list_predictive_edge_runs(
    session: AsyncSession,
    *,
    asset: str | None = None,
    provider: str | None = None,
    market_type: str | None = None,
    interval: str | None = None,
    status: str | None = None,
    cursor: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> PredictiveEdgeRunPageResponse:
    statement = select(PredictiveEdgeRunRecord)
    for column, value in (
        (PredictiveEdgeRunRecord.asset, asset),
        (PredictiveEdgeRunRecord.provider, provider),
        (PredictiveEdgeRunRecord.market_type, market_type),
        (PredictiveEdgeRunRecord.interval, interval),
        (PredictiveEdgeRunRecord.status, status),
    ):
        if value is not None:
            statement = statement.where(column == value)
    if cursor:
        try:
            timestamp_text, identifier = cursor.rsplit("|", 1)
            timestamp = datetime.fromisoformat(timestamp_text)
            run_id = UUID(identifier)
        except ValueError as error:
            raise PredictiveEdgeReadError("invalid run cursor") from error
        statement = statement.where(
            or_(
                PredictiveEdgeRunRecord.started_at < timestamp,
                and_(
                    PredictiveEdgeRunRecord.started_at == timestamp,
                    PredictiveEdgeRunRecord.id < run_id,
                ),
            )
        )
    rows = (
        (
            await session.scalars(
                statement.order_by(
                    PredictiveEdgeRunRecord.started_at.desc(),
                    PredictiveEdgeRunRecord.id.desc(),
                ).limit(_limit(limit) + 1)
            )
        )
        .unique()
        .all()
    )
    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = (
        f"{page[-1].started_at.isoformat()}|{page[-1].id}"
        if has_more and page
        else None
    )
    return PredictiveEdgeRunPageResponse(
        items=tuple(_run(item) for item in page),
        next_cursor=next_cursor,
        has_more=has_more,
    )


async def get_predictive_edge_run(
    session: AsyncSession, run_id: UUID
) -> PredictiveEdgeRunDetailResponse:
    run = await session.get(PredictiveEdgeRunRecord, run_id)
    if run is None:
        raise PredictiveEdgeNotFoundError(f"predictive-edge run {run_id} was not found")
    return PredictiveEdgeRunDetailResponse(
        run=_run(run),
        configuration={
            "fee_rate": run.fee_rate,
            "slippage_rate": run.slippage_rate,
            "neutral_threshold": run.neutral_threshold,
            "decision_interval_seconds": run.decision_interval_seconds,
            "training_days": run.training_days,
            "validation_days": run.validation_days,
            "test_days": run.test_days,
            "step_days": run.step_days,
            "maximum_windows": run.maximum_windows,
            "bin_count": run.bin_count,
            "minimum_observations": run.minimum_observations,
            "minimum_windows": run.minimum_windows,
            "minimum_hit_rate_improvement_over_benchmark": (
                run.minimum_hit_rate_improvement_over_benchmark
            ),
        },
    )


async def list_predictive_edges(
    session: AsyncSession,
    *,
    run_id: UUID,
    predictor: str | None = None,
    horizon_seconds: int | None = None,
    condition: str | None = None,
    classification: EvidenceStatus | None = None,
    cursor: UUID | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> PredictiveEdgePageResponse:
    await _require_run(session, run_id)
    statement = select(PredictiveEdgeAggregateRecord).where(
        PredictiveEdgeAggregateRecord.run_id == run_id
    )
    for column, value in (
        (PredictiveEdgeAggregateRecord.predictor_name, predictor),
        (PredictiveEdgeAggregateRecord.horizon_seconds, horizon_seconds),
        (PredictiveEdgeAggregateRecord.condition, condition),
        (
            PredictiveEdgeAggregateRecord.classification,
            classification.value if classification else None,
        ),
    ):
        if value is not None:
            statement = statement.where(column == value)
    if cursor is not None:
        statement = statement.where(PredictiveEdgeAggregateRecord.id > cursor)
    rows = (
        await session.scalars(
            statement.order_by(PredictiveEdgeAggregateRecord.id.asc()).limit(
                _limit(limit) + 1
            )
        )
    ).all()
    has_more = len(rows) > limit
    page = rows[:limit]
    return PredictiveEdgePageResponse(
        items=tuple(
            PredictiveEdgeResponse.model_validate(item, from_attributes=True)
            for item in page
        ),
        next_cursor=str(page[-1].id) if has_more and page else None,
        has_more=has_more,
    )


async def list_predictive_edge_windows(
    session: AsyncSession,
    *,
    run_id: UUID,
    edge_id: UUID,
    cursor: int | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> PredictiveEdgeWindowPageResponse:
    edge = await session.get(PredictiveEdgeAggregateRecord, edge_id)
    if edge is None or edge.run_id != run_id:
        raise PredictiveEdgeNotFoundError(f"predictive edge {edge_id} was not found")
    statement = (
        select(PredictiveEdgeWindowMetricRecord, PredictiveEdgeWindowRecord.ordinal)
        .join(
            PredictiveEdgeWindowRecord,
            PredictiveEdgeWindowRecord.id == PredictiveEdgeWindowMetricRecord.window_id,
        )
        .where(
            PredictiveEdgeWindowMetricRecord.run_id == run_id,
            PredictiveEdgeWindowMetricRecord.predictor_name == edge.predictor_name,
            PredictiveEdgeWindowMetricRecord.horizon_seconds == edge.horizon_seconds,
            PredictiveEdgeWindowMetricRecord.condition == edge.condition,
            PredictiveEdgeWindowMetricRecord.direction_mapping
            == edge.direction_mapping,
        )
    )
    if cursor is not None:
        statement = statement.where(PredictiveEdgeWindowRecord.ordinal > cursor)
    rows = (
        await session.execute(
            statement.order_by(PredictiveEdgeWindowRecord.ordinal.asc()).limit(
                _limit(limit) + 1
            )
        )
    ).all()
    has_more = len(rows) > limit
    page = rows[:limit]
    metric_ids = tuple(value.id for value, _ in page)
    bin_rows = (
        (
            await session.scalars(
                select(PredictiveEdgeBinAggregateRecord)
                .where(
                    PredictiveEdgeBinAggregateRecord.window_metric_id.in_(metric_ids)
                )
                .order_by(
                    PredictiveEdgeBinAggregateRecord.window_metric_id.asc(),
                    PredictiveEdgeBinAggregateRecord.ordinal.asc(),
                )
            )
        ).all()
        if metric_ids
        else []
    )
    bins_by_metric: dict[UUID, list[PredictiveEdgeBinAggregateRecord]] = {}
    for bin_row in bin_rows:
        bins_by_metric.setdefault(bin_row.window_metric_id, []).append(bin_row)
    return PredictiveEdgeWindowPageResponse(
        items=tuple(
            PredictiveEdgeWindowMetricResponse(
                id=value.id,
                window_ordinal=ordinal,
                semantic_kind=PredictorSemanticKind(value.semantic_kind),
                training_center=value.training_center,
                direction_mapping=DirectionMapping(value.direction_mapping),
                validation_status=ValidationStatus(value.validation_status),
                validation_direction=DirectionMapping(value.validation_direction),
                validation_usable_count=value.validation_usable_count,
                validation_correlation=value.validation_correlation,
                validation_directional_hit_rate=(value.validation_directional_hit_rate),
                validation_mean_strategy_aligned_net_return=(
                    value.validation_mean_strategy_aligned_net_return
                ),
                selected_for_test=value.selected_for_test,
                selection_reason=value.selection_reason,
                selected_bin_ordinals=tuple(value.selected_bin_ordinals),
                test_confirmation_status=TestConfirmationStatus(
                    value.test_confirmation_status
                ),
                test_usable_count=value.test_usable_count,
                test_missing_count=value.test_missing_count,
                test_correlation=value.test_correlation,
                test_directional_hit_rate=value.test_directional_hit_rate,
                mean_forward_gross_return=value.mean_forward_gross_return,
                mean_absolute_movement=value.mean_absolute_movement,
                mean_net_movement_magnitude_after_friction=(
                    value.mean_net_movement_magnitude_after_friction
                ),
                positive_movement_after_friction_ratio=(
                    value.positive_movement_after_friction_ratio
                ),
                mean_strategy_aligned_gross_return=(
                    value.mean_strategy_aligned_gross_return
                ),
                mean_strategy_aligned_net_return=(
                    value.mean_strategy_aligned_net_return
                ),
                mean_expected_avoided_downside=(value.mean_expected_avoided_downside),
                test_standard_error=value.test_standard_error,
                test_evidence_status=EvidenceStatus(value.test_evidence_status),
                test_insufficient_reason=value.test_insufficient_reason,
                training_bin_bounds=tuple(
                    Decimal(item)
                    for item in value.training_bin_bounds.split(",")
                    if item
                ),
                volatility_threshold=value.volatility_threshold,
                bin_selections=tuple(
                    PredictiveEdgeBinSelectionResponse(
                        bin_ordinal=item.ordinal,
                        lower_bound=item.lower_bound,
                        upper_bound=item.upper_bound,
                        suggested_action=SuggestedAction(item.suggested_action),
                        validation_observation_count=(
                            item.validation_observation_count
                        ),
                        validation_gross_return=item.validation_gross_return,
                        validation_long_aligned_net_return=(
                            item.validation_long_aligned_net_return
                        ),
                        validation_avoided_downside=(item.validation_avoided_downside),
                        validation_directional_hit_rate=(
                            item.validation_directional_hit_rate
                        ),
                        selected_for_test=item.selected_for_test,
                        test_observation_count=item.test_observation_count,
                        test_confirmation_status=TestConfirmationStatus(
                            item.test_confirmation_status
                        ),
                        test_gross_return=item.test_gross_return,
                        test_long_aligned_net_return=(
                            item.test_long_aligned_net_return
                        ),
                        test_avoided_downside=item.test_avoided_downside,
                        test_directional_hit_rate=item.test_directional_hit_rate,
                        rejection_reason=item.rejection_reason,
                    )
                    for item in bins_by_metric.get(value.id, ())
                    if item.ordinal == edge.bin_ordinal
                    and item.suggested_action == edge.suggested_action
                ),
            )
            for value, ordinal in page
        ),
        next_cursor=str(page[-1][1]) if has_more and page else None,
        has_more=has_more,
    )


async def list_strategy_hypotheses(
    session: AsyncSession,
    *,
    run_id: UUID,
    cursor: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> StrategyHypothesisPageResponse:
    await _require_run(session, run_id)
    statement = select(StrategyHypothesisRecord).where(
        StrategyHypothesisRecord.run_id == run_id
    )
    if cursor is not None:
        statement = statement.where(StrategyHypothesisRecord.hypothesis_id > cursor)
    rows = (
        await session.scalars(
            statement.order_by(StrategyHypothesisRecord.hypothesis_id.asc()).limit(
                _limit(limit) + 1
            )
        )
    ).all()
    has_more = len(rows) > limit
    page = rows[:limit]
    return StrategyHypothesisPageResponse(
        items=tuple(
            StrategyHypothesisResponse.model_validate(item, from_attributes=True)
            for item in page
        ),
        next_cursor=page[-1].hypothesis_id if has_more and page else None,
        has_more=has_more,
    )


async def _require_run(session: AsyncSession, run_id: UUID) -> None:
    if await session.get(PredictiveEdgeRunRecord, run_id) is None:
        raise PredictiveEdgeNotFoundError(f"predictive-edge run {run_id} was not found")


def _run(value: PredictiveEdgeRunRecord) -> PredictiveEdgeRunResponse:
    return PredictiveEdgeRunResponse.model_validate(value, from_attributes=True)


async def get_predictive_edge_diagnostics(
    session: AsyncSession, run_id: UUID
) -> PredictiveEdgeDiagnosticResponse:
    await _require_run(session, run_id)
    record = await session.scalar(
        select(PredictiveEdgeRunDiagnosticRecord).where(
            PredictiveEdgeRunDiagnosticRecord.run_id == run_id
        )
    )
    if record is None:
        raise PredictiveEdgeNotFoundError(
            f"predictive-edge run {run_id} has no v4 diagnostics"
        )
    return PredictiveEdgeDiagnosticResponse.model_validate(record, from_attributes=True)


def _near_miss(value: PredictiveEdgeNearMissRecord) -> PredictiveEdgeNearMissResponse:
    return PredictiveEdgeNearMissResponse.model_validate(value, from_attributes=True)


async def list_predictive_edge_near_misses(
    session: AsyncSession,
    *,
    run_id: UUID,
    cursor: int | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
) -> PredictiveEdgeNearMissPageResponse:
    await _require_run(session, run_id)
    statement = select(PredictiveEdgeNearMissRecord).where(
        PredictiveEdgeNearMissRecord.run_id == run_id
    )
    if cursor is not None:
        statement = statement.where(PredictiveEdgeNearMissRecord.rank > cursor)
    rows = (
        await session.scalars(
            statement.order_by(PredictiveEdgeNearMissRecord.rank.asc()).limit(
                _limit(limit) + 1
            )
        )
    ).all()
    has_more = len(rows) > limit
    page = rows[:limit]
    return PredictiveEdgeNearMissPageResponse(
        items=tuple(_near_miss(item) for item in page),
        next_cursor=page[-1].rank if has_more and page else None,
        has_more=has_more,
    )


async def get_predictive_edge_near_miss(
    session: AsyncSession, *, run_id: UUID, near_miss_id: UUID
) -> PredictiveEdgeNearMissDetailResponse:
    await _require_run(session, run_id)
    record = await session.scalar(
        select(PredictiveEdgeNearMissRecord).where(
            PredictiveEdgeNearMissRecord.id == near_miss_id,
            PredictiveEdgeNearMissRecord.run_id == run_id,
        )
    )
    if record is None:
        raise PredictiveEdgeNotFoundError(
            f"predictive-edge near miss {near_miss_id} was not found"
        )
    sensitivity = (
        await session.scalars(
            select(PredictiveEdgeSensitivityRecord)
            .where(
                PredictiveEdgeSensitivityRecord.run_id == run_id,
                PredictiveEdgeSensitivityRecord.edge_fingerprint
                == record.edge_fingerprint,
            )
            .order_by(PredictiveEdgeSensitivityRecord.threshold_name.asc())
        )
    ).all()
    return PredictiveEdgeNearMissDetailResponse(
        near_miss=_near_miss(record),
        sensitivity=tuple(
            ThresholdSensitivityResponse.model_validate(item, from_attributes=True)
            for item in sensitivity
        ),
    )
