from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Select, String, and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import temporal_fold_definitions
from app.ml.screening.config import ScreeningEngine
from app.ml.screening.evidence import (
    EVIDENCE_CLASSIFICATION_VERSION,
    EvidenceClassification,
    classify_evidence,
)
from app.ml.screening.models import (
    MLModelScreeningEvaluationRecord,
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)
from app.ml.screening.schemas import (
    ClassificationCounts,
    FoldEvaluationResponse,
    ScreeningResultDetail,
    ScreeningResultPage,
    ScreeningResultResponse,
    ScreeningRunDetail,
    ScreeningRunPage,
    ScreeningRunSummary,
)

MAX_SCREENING_PAGE_LIMIT = 200


class ScreeningReadNotFoundError(LookupError):
    pass


def _classification_expression() -> ColumnElement[str]:
    row = MLModelScreeningResultRecord
    incomplete = (
        (row.folds_evaluated != 7)
        | row.median_pearson_correlation.is_(None)
        | row.median_spearman_correlation.is_(None)
    )
    promising = and_(
        row.folds_evaluated == 7,
        row.folds_beating_zero == 7,
        row.folds_with_positive_spearman == 7,
        row.median_mae_improvement_over_zero > 0,
        row.worst_fold_mae_improvement_over_zero > 0,
        row.median_directional_accuracy > 0.5,
        row.worst_fold_directional_accuracy >= 0.5,
        row.median_spearman_correlation > 0,
        row.median_pearson_correlation > 0,
    )
    partial = (
        (
            (row.median_mae_improvement_over_zero > 0)
            & (row.median_spearman_correlation > 0)
        )
        | (
            (row.median_mae_improvement_over_zero > 0)
            & (row.median_directional_accuracy > 0.5)
        )
        | (
            (row.median_spearman_correlation > 0)
            & (row.median_directional_accuracy > 0.5)
        )
    )
    return case(
        (incomplete, EvidenceClassification.INCOMPLETE.value),
        (promising, EvidenceClassification.PROMISING.value),
        (partial, EvidenceClassification.INSUFFICIENT.value),
        else_=EvidenceClassification.DISCARDED.value,
    ).cast(String)


def _result_response(row: MLModelScreeningResultRecord) -> ScreeningResultResponse:
    assessment = classify_evidence(row)
    values = {
        column.name: getattr(row, column.name) for column in row.__table__.columns
    }
    return ScreeningResultResponse(
        **values,
        evidence_classification=assessment.classification,
        evidence_classification_version=EVIDENCE_CLASSIFICATION_VERSION,
        classification_explanation=assessment.explanation,
        failed_classification_rules=list(assessment.failed_rules),
    )


def _run_filters(
    statement: Select[tuple[MLModelScreeningRunRecord]],
    *,
    asset: Asset | None,
    provider: Provider | None,
    market_type: MarketType | None,
    interval: Interval | None,
    status: str | None,
) -> Select[tuple[MLModelScreeningRunRecord]]:
    for column, value in (
        (MLModelScreeningRunRecord.asset, asset),
        (MLModelScreeningRunRecord.provider, provider),
        (MLModelScreeningRunRecord.market_type, market_type),
        (MLModelScreeningRunRecord.interval, interval),
        (MLModelScreeningRunRecord.status, status),
    ):
        if value is not None:
            statement = statement.where(column == value)
    return statement


async def list_screening_runs(
    session: AsyncSession,
    *,
    asset: Asset | None = None,
    provider: Provider | None = None,
    market_type: MarketType | None = None,
    interval: Interval | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ScreeningRunPage:
    base = _run_filters(
        select(MLModelScreeningRunRecord),
        asset=asset,
        provider=provider,
        market_type=market_type,
        interval=interval,
        status=status,
    )
    total = await session.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = (
        await session.scalars(
            base.order_by(
                MLModelScreeningRunRecord.started_at.desc(),
                MLModelScreeningRunRecord.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return ScreeningRunPage(
        items=[ScreeningRunSummary.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(rows) < total,
    )


async def latest_completed_screening_run(
    session: AsyncSession,
    *,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
) -> ScreeningRunSummary:
    row = await session.scalar(
        select(MLModelScreeningRunRecord)
        .where(
            MLModelScreeningRunRecord.asset == asset,
            MLModelScreeningRunRecord.provider == provider,
            MLModelScreeningRunRecord.market_type == market_type,
            MLModelScreeningRunRecord.interval == interval,
            MLModelScreeningRunRecord.status == "completed",
        )
        .order_by(
            MLModelScreeningRunRecord.completed_at.desc(),
            MLModelScreeningRunRecord.id.desc(),
        )
        .limit(1)
    )
    if row is None:
        raise ScreeningReadNotFoundError("completed ML screening run not found")
    return ScreeningRunSummary.model_validate(row)


async def get_screening_run(
    session: AsyncSession, run_id: UUID
) -> MLModelScreeningRunRecord:
    row = await session.get(MLModelScreeningRunRecord, run_id)
    if row is None:
        raise ScreeningReadNotFoundError(f"ML screening run {run_id} not found")
    return row


async def get_screening_run_detail(
    session: AsyncSession, run_id: UUID
) -> ScreeningRunDetail:
    run = await get_screening_run(session, run_id)
    classification = _classification_expression()
    class_rows = (
        await session.execute(
            select(classification, func.count())
            .where(MLModelScreeningResultRecord.run_id == run_id)
            .group_by(classification)
        )
    ).all()
    counts = {item.value: 0 for item in EvidenceClassification}
    counts.update({str(name): count for name, count in class_rows})
    engine_rows = (
        await session.execute(
            select(MLModelScreeningResultRecord.engine, func.count())
            .where(MLModelScreeningResultRecord.run_id == run_id)
            .group_by(MLModelScreeningResultRecord.engine)
        )
    ).all()
    return ScreeningRunDetail(
        run=ScreeningRunSummary.model_validate(run),
        classification_counts=ClassificationCounts(**counts),
        engine_counts={ScreeningEngine(name): count for name, count in engine_rows},
        pareto_retained_count=run.pareto_retained_count,
    )


async def list_screening_results(
    session: AsyncSession,
    run_id: UUID,
    *,
    engine: ScreeningEngine | None = None,
    pareto_retained: bool | None = None,
    evidence_classification: EvidenceClassification | None = None,
    minimum_future_duration_seconds: int | None = None,
    maximum_future_duration_seconds: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ScreeningResultPage:
    await get_screening_run(session, run_id)
    classification = _classification_expression()
    query = select(MLModelScreeningResultRecord).where(
        MLModelScreeningResultRecord.run_id == run_id
    )
    for column, value in (
        (MLModelScreeningResultRecord.engine, engine),
        (MLModelScreeningResultRecord.pareto_retained, pareto_retained),
    ):
        if value is not None:
            query = query.where(column == value)
    if evidence_classification is not None:
        query = query.where(classification == evidence_classification.value)
    if minimum_future_duration_seconds is not None:
        query = query.where(
            MLModelScreeningResultRecord.future_duration_seconds
            >= minimum_future_duration_seconds
        )
    if maximum_future_duration_seconds is not None:
        query = query.where(
            MLModelScreeningResultRecord.future_duration_seconds
            <= maximum_future_duration_seconds
        )
    total = (
        await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    )
    priority = case(
        {value.value: index for index, value in enumerate(EvidenceClassification)},
        value=classification,
    )
    rows = (
        await session.scalars(
            query.order_by(
                priority,
                MLModelScreeningResultRecord.pareto_retained.desc(),
                MLModelScreeningResultRecord.display_rank,
                MLModelScreeningResultRecord.engine,
                MLModelScreeningResultRecord.future_duration_seconds,
                MLModelScreeningResultRecord.id,
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return ScreeningResultPage(
        items=[_result_response(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(rows) < total,
    )


async def get_screening_result_detail(
    session: AsyncSession, run_id: UUID, result_id: UUID
) -> ScreeningResultDetail:
    await get_screening_run(session, run_id)
    result = await session.scalar(
        select(MLModelScreeningResultRecord).where(
            MLModelScreeningResultRecord.id == result_id,
            MLModelScreeningResultRecord.run_id == run_id,
        )
    )
    if result is None:
        raise ScreeningReadNotFoundError(
            f"ML screening result {result_id} not found for run {run_id}"
        )
    order = {fold.name: index for index, fold in enumerate(temporal_fold_definitions())}
    fold_order = case(
        order, value=MLModelScreeningEvaluationRecord.fold_name, else_=len(order)
    )
    folds: Sequence[MLModelScreeningEvaluationRecord] = (
        await session.scalars(
            select(MLModelScreeningEvaluationRecord)
            .where(
                MLModelScreeningEvaluationRecord.run_id == run_id,
                MLModelScreeningEvaluationRecord.engine == result.engine,
                MLModelScreeningEvaluationRecord.future_duration_seconds
                == result.future_duration_seconds,
            )
            .order_by(fold_order, MLModelScreeningEvaluationRecord.id)
        )
    ).all()
    return ScreeningResultDetail(
        result=_result_response(result),
        fold_evaluations=[FoldEvaluationResponse.model_validate(row) for row in folds],
    )
