from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.screening.config import ScreeningEngine

from .contracts import PredictionObjective, RefinementClassification, RefinementStatus
from .models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from .schemas import (
    RefinementEvaluationDetail,
    RefinementEvaluationPage,
    RefinementEvaluationResponse,
    RefinementRunDetail,
    RefinementRunPage,
    RefinementRunResponse,
)

MAX_REFINEMENT_PAGE_LIMIT = 200


class RefinementReadNotFoundError(LookupError):
    pass


async def get_run_record(session: AsyncSession, run_id: UUID) -> MLRefinementRunRecord:
    record = await session.get(MLRefinementRunRecord, run_id)
    if record is None:
        raise RefinementReadNotFoundError(f"ML refinement run {run_id} not found")
    return record


async def list_runs(
    session: AsyncSession,
    *,
    status: RefinementStatus | None,
    limit: int,
    offset: int,
) -> RefinementRunPage:
    query = select(MLRefinementRunRecord)
    if status is not None:
        query = query.where(MLRefinementRunRecord.status == status.value)
    total = (
        await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    )
    rows = (
        await session.scalars(
            query.order_by(
                MLRefinementRunRecord.created_at.desc(),
                MLRefinementRunRecord.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return RefinementRunPage(
        items=[RefinementRunResponse.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(rows) < total,
    )


async def get_run_detail(session: AsyncSession, run_id: UUID) -> RefinementRunDetail:
    record = await get_run_record(session, run_id)
    classifications = {item.value: 0 for item in RefinementClassification}
    for name, count in (
        await session.execute(
            select(MLRefinementEvaluationRecord.classification, func.count())
            .where(MLRefinementEvaluationRecord.run_id == run_id)
            .group_by(MLRefinementEvaluationRecord.classification)
        )
    ).all():
        classifications[name] = count
    engines = {
        ScreeningEngine(name): count
        for name, count in (
            await session.execute(
                select(MLRefinementEvaluationRecord.engine, func.count())
                .where(MLRefinementEvaluationRecord.run_id == run_id)
                .group_by(MLRefinementEvaluationRecord.engine)
            )
        ).all()
    }
    return RefinementRunDetail(
        run=RefinementRunResponse.model_validate(record),
        classification_counts=classifications,
        engine_counts=engines,
    )


async def list_evaluations(
    session: AsyncSession,
    run_id: UUID,
    *,
    engine: ScreeningEngine | None,
    objective: PredictionObjective | None,
    classification: RefinementClassification | None,
    limit: int,
    offset: int,
) -> RefinementEvaluationPage:
    await get_run_record(session, run_id)
    query = select(MLRefinementEvaluationRecord).where(
        MLRefinementEvaluationRecord.run_id == run_id
    )
    for column, value in (
        (MLRefinementEvaluationRecord.engine, engine.value if engine else None),
        (
            MLRefinementEvaluationRecord.objective,
            objective.value if objective else None,
        ),
        (
            MLRefinementEvaluationRecord.classification,
            classification.value if classification else None,
        ),
    ):
        if value is not None:
            query = query.where(column == value)
    total = (
        await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    )
    rows = (
        await session.scalars(
            query.order_by(
                MLRefinementEvaluationRecord.classification,
                MLRefinementEvaluationRecord.engine,
                MLRefinementEvaluationRecord.future_duration_seconds,
                MLRefinementEvaluationRecord.fingerprint,
                MLRefinementEvaluationRecord.id,
            )
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return RefinementEvaluationPage(
        items=[RefinementEvaluationResponse.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(rows) < total,
    )


async def get_evaluation(
    session: AsyncSession, run_id: UUID, evaluation_id: UUID
) -> RefinementEvaluationDetail:
    await get_run_record(session, run_id)
    record = await session.scalar(
        select(MLRefinementEvaluationRecord).where(
            MLRefinementEvaluationRecord.id == evaluation_id,
            MLRefinementEvaluationRecord.run_id == run_id,
        )
    )
    if record is None:
        raise RefinementReadNotFoundError(
            f"ML refinement evaluation {evaluation_id} not found for run {run_id}"
        )
    return RefinementEvaluationDetail.model_validate(record)
