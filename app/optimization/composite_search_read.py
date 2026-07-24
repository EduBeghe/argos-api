from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.optimization.composite_search_memory import (
    CompositeSearchCandidateRecord,
    CompositeSearchRunRecord,
)
from app.optimization.composite_search_schemas import (
    CompositeCandidateDetailResponse,
    CompositeCandidatePageResponse,
    CompositeCandidateResponse,
    CompositeSearchSummaryResponse,
)

MAX_COMPOSITE_PAGE_LIMIT = 100


class CompositeSearchNotFoundError(LookupError):
    """Raised when persisted composite evidence does not exist."""


async def get_composite_search(
    session: AsyncSession, predictive_edge_run_id: UUID
) -> CompositeSearchSummaryResponse:
    run = await session.scalar(
        select(CompositeSearchRunRecord).where(
            CompositeSearchRunRecord.predictive_edge_run_id == predictive_edge_run_id
        )
    )
    if run is None:
        raise CompositeSearchNotFoundError("composite search was not found")
    confirmed = await session.scalar(
        select(func.count())
        .select_from(CompositeSearchCandidateRecord)
        .where(
            CompositeSearchCandidateRecord.run_id == run.id,
            CompositeSearchCandidateRecord.status == "confirmed_candidate",
        )
    )
    return CompositeSearchSummaryResponse(
        id=run.id,
        predictive_edge_run_id=run.predictive_edge_run_id,
        conclusion=run.conclusion,
        maximum_depth_reached=run.maximum_depth_reached,
        feature_pool=tuple(run.feature_pool),
        excluded_features=run.excluded_features,
        funnel_by_depth=tuple(run.funnel_by_depth),
        total_evaluations=run.total_evaluations,
        computed=run.computed,
        reused=run.reused,
        best_depth=run.best_depth,
        best_candidate_fingerprint=run.best_candidate_fingerprint,
        confirmed_candidate_count=int(confirmed or 0),
        dominant_rejection_reason=run.dominant_rejection_reason,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


def _candidate(value: CompositeSearchCandidateRecord) -> CompositeCandidateResponse:
    return CompositeCandidateResponse.model_validate(value, from_attributes=True)


async def list_composite_candidates(
    session: AsyncSession,
    predictive_edge_run_id: UUID,
    *,
    depth: int | None,
    status: str | None,
    feature_name: str | None,
    limit: int,
    offset: int,
) -> CompositeCandidatePageResponse:
    run = await session.scalar(
        select(CompositeSearchRunRecord).where(
            CompositeSearchRunRecord.predictive_edge_run_id == predictive_edge_run_id
        )
    )
    if run is None:
        raise CompositeSearchNotFoundError("composite search was not found")
    filters = [CompositeSearchCandidateRecord.run_id == run.id]
    if depth is not None:
        filters.append(CompositeSearchCandidateRecord.depth == depth)
    if status is not None:
        filters.append(CompositeSearchCandidateRecord.status == status)
    if feature_name is not None:
        filters.append(CompositeSearchCandidateRecord.features.contains([feature_name]))
    total = int(
        await session.scalar(
            select(func.count())
            .select_from(CompositeSearchCandidateRecord)
            .where(*filters)
        )
        or 0
    )
    records = tuple(
        (
            await session.scalars(
                select(CompositeSearchCandidateRecord)
                .where(*filters)
                .order_by(
                    CompositeSearchCandidateRecord.depth,
                    CompositeSearchCandidateRecord.candidate_fingerprint,
                )
                .limit(limit)
                .offset(offset)
            )
        ).all()
    )
    return CompositeCandidatePageResponse(
        items=tuple(_candidate(value) for value in records),
        total=total,
        limit=limit,
        offset=offset,
    )


async def get_composite_candidate(
    session: AsyncSession, predictive_edge_run_id: UUID, candidate_id: UUID
) -> CompositeCandidateDetailResponse:
    record = await session.scalar(
        select(CompositeSearchCandidateRecord)
        .join(CompositeSearchRunRecord)
        .where(
            CompositeSearchRunRecord.predictive_edge_run_id == predictive_edge_run_id,
            CompositeSearchCandidateRecord.id == candidate_id,
        )
    )
    if record is None:
        raise CompositeSearchNotFoundError("composite candidate was not found")
    return CompositeCandidateDetailResponse(
        run_id=record.run_id, **_candidate(record).model_dump()
    )
