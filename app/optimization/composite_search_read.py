from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.optimization.composite_search import (
    CompositeCandidateStatus,
    CompositeSelectionTrack,
)
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
        budget_exhausted=run.budget_exhausted,
        cheap_evaluations_computed=run.cheap_evaluations_computed,
        cheap_evaluations_reused=run.cheap_evaluations_reused,
        training_simulations_computed=run.training_simulations_computed,
        training_simulations_reused=run.training_simulations_reused,
        validation_simulations_computed=run.validation_simulations_computed,
        validation_simulations_reused=run.validation_simulations_reused,
        test_simulations_computed=run.test_simulations_computed,
        test_simulations_reused=run.test_simulations_reused,
        best_depth=run.best_depth,
        best_candidate_fingerprint=run.best_candidate_fingerprint,
        confirmed_candidate_count=int(confirmed or 0),
        dominant_rejection_reason=run.dominant_rejection_reason,
        best_economic_candidate_fingerprint=run.best_economic_candidate_fingerprint,
        best_exploratory_candidate_fingerprint=(
            run.best_exploratory_candidate_fingerprint
        ),
        candidate_simulations_computed=run.candidate_simulations_computed,
        candidate_simulations_reused=run.candidate_simulations_reused,
        production_simulations_computed=run.production_simulations_computed,
        production_simulations_reused=run.production_simulations_reused,
        buy_and_hold_simulations_computed=run.buy_and_hold_simulations_computed,
        buy_and_hold_simulations_reused=run.buy_and_hold_simulations_reused,
        normalizations_computed=run.normalizations_computed,
        normalizations_reused=run.normalizations_reused,
        simulations_avoided=run.simulations_avoided,
        row_slices_computed=run.row_slices_computed,
        row_slices_reused=run.row_slices_reused,
        threshold_fits_computed=run.threshold_fits_computed,
        threshold_fits_reused=run.threshold_fits_reused,
        persistence_truncated=run.persistence_truncated,
        depth_quota_reached=run.depth_quota_reached,
        global_compute_budget_exhausted=run.global_compute_budget_exhausted,
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
    status: CompositeCandidateStatus | None,
    selection_track: CompositeSelectionTrack | None,
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
    if selection_track is not None:
        filters.append(
            CompositeSearchCandidateRecord.selection_track == selection_track
        )
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
