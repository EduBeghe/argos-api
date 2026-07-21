from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.optimization.predictive_edge_memory import (
    PredictiveEdgeAggregateRecord,
    PredictiveEdgeRunRecord,
    PredictiveEdgeWindowMetricRecord,
    PredictiveEdgeWindowRecord,
    StrategyHypothesisRecord,
)
from app.optimization.predictive_edge_schemas import (
    PredictiveEdgePageResponse,
    PredictiveEdgeResponse,
    PredictiveEdgeRunDetailResponse,
    PredictiveEdgeRunPageResponse,
    PredictiveEdgeRunResponse,
    PredictiveEdgeWindowMetricResponse,
    PredictiveEdgeWindowPageResponse,
    StrategyHypothesisPageResponse,
    StrategyHypothesisResponse,
)
from app.optimization.predictive_edges import EvidenceStatus

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
    return PredictiveEdgeWindowPageResponse(
        items=tuple(
            PredictiveEdgeWindowMetricResponse(
                id=value.id,
                window_ordinal=ordinal,
                usable_count=value.usable_count,
                missing_count=value.missing_count,
                correlation=value.correlation,
                directional_hit_rate=value.directional_hit_rate,
                mean_gross_return=value.mean_gross_return,
                mean_net_return=value.mean_net_return,
                positive_net_ratio=value.positive_net_ratio,
                standard_error=value.standard_error,
                evidence_status=EvidenceStatus(value.evidence_status),
                insufficient_reason=value.insufficient_reason,
                training_bin_bounds=tuple(
                    Decimal(item)
                    for item in value.training_bin_bounds.split(",")
                    if item
                ),
                volatility_threshold=value.volatility_threshold,
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

