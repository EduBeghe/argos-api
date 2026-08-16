from uuid import UUID

from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.optimization._pagination import count_matching
from app.optimization._persistence_helpers import require
from app.optimization.targeted_refinement import TargetedCandidateStatus
from app.optimization.targeted_refinement_memory import (
    TargetedRefinementCandidateRecord,
    TargetedRefinementRunRecord,
)
from app.optimization.targeted_refinement_schemas import (
    TargetedCandidateDetailResponse,
    TargetedCandidatePageResponse,
    TargetedCandidateResponse,
    TargetedRefinementSummaryResponse,
)

MAX_TARGETED_PAGE_LIMIT = 100


class TargetedRefinementNotFoundError(LookupError):
    """Raised when targeted evidence does not exist."""


async def get_targeted_refinement(
    session: AsyncSession, predictive_edge_run_id: UUID
) -> TargetedRefinementSummaryResponse:
    row = require(
        await session.scalar(
            select(TargetedRefinementRunRecord).where(
                TargetedRefinementRunRecord.predictive_edge_run_id
                == predictive_edge_run_id
            )
        ),
        TargetedRefinementNotFoundError("targeted refinement was not found"),
    )
    confirmed = list(
        (
            await session.scalars(
                select(TargetedRefinementCandidateRecord)
                .where(
                    TargetedRefinementCandidateRecord.run_id == row.id,
                    TargetedRefinementCandidateRecord.status
                    == TargetedCandidateStatus.CONFIRMED.value,
                )
                .order_by(
                    TargetedRefinementCandidateRecord.normalized_distance,
                    TargetedRefinementCandidateRecord.candidate_fingerprint,
                )
            )
        ).all()
    )
    best = confirmed[0] if confirmed else None
    return TargetedRefinementSummaryResponse(
        id=row.id,
        predictive_edge_run_id=row.predictive_edge_run_id,
        conclusion=row.conclusion,
        summary=row.summary,
        confirmed_candidate_count=len(confirmed),
        best_targeted_candidate_id=best.id if best is not None else None,
        best_targeted_candidate=(
            best.candidate_fingerprint if best is not None else None
        ),
    )


def _response(row: TargetedRefinementCandidateRecord) -> TargetedCandidateResponse:
    return TargetedCandidateResponse(
        id=row.id,
        candidate_fingerprint=row.candidate_fingerprint,
        seed_fingerprint=row.seed_fingerprint,
        status=row.status,
        normalized_distance=row.normalized_distance,
        configuration=row.configuration,
        training_metrics=row.training_metrics,
        validation_metrics=row.validation_metrics,
        test_metrics=row.test_metrics,
        passed_requirements=tuple(row.passed_requirements),
        failed_requirements=tuple(row.failed_requirements),
        requirement_margins=tuple(row.requirement_margins),
        rejection_reason=row.rejection_reason,
    )


async def list_targeted_candidates(
    session: AsyncSession,
    predictive_edge_run_id: UUID,
    *,
    status: TargetedCandidateStatus | None,
    limit: int,
    offset: int,
) -> TargetedCandidatePageResponse:
    run = require(
        await session.scalar(
            select(TargetedRefinementRunRecord.id).where(
                TargetedRefinementRunRecord.predictive_edge_run_id
                == predictive_edge_run_id
            )
        ),
        TargetedRefinementNotFoundError("targeted refinement was not found"),
    )
    filters = [TargetedRefinementCandidateRecord.run_id == run]
    if status is not None:
        filters.append(TargetedRefinementCandidateRecord.status == status.value)
    total = await count_matching(session, TargetedRefinementCandidateRecord, *filters)
    priority = case(
        (
            TargetedRefinementCandidateRecord.status
            == TargetedCandidateStatus.CONFIRMED.value,
            0,
        ),
        else_=1,
    )
    rows = (
        await session.scalars(
            select(TargetedRefinementCandidateRecord)
            .where(*filters)
            .order_by(
                priority,
                TargetedRefinementCandidateRecord.normalized_distance,
                TargetedRefinementCandidateRecord.candidate_fingerprint,
            )
            .offset(offset)
            .limit(limit)
        )
    ).all()
    next_offset = offset + len(rows)
    return TargetedCandidatePageResponse(
        items=tuple(_response(row) for row in rows),
        total=total,
        next_cursor=str(next_offset) if next_offset < total else None,
        has_more=next_offset < total,
    )


async def get_targeted_candidate(
    session: AsyncSession, predictive_edge_run_id: UUID, candidate_id: UUID
) -> TargetedCandidateDetailResponse:
    row = require(
        await session.scalar(
            select(TargetedRefinementCandidateRecord)
            .join(
                TargetedRefinementRunRecord,
                TargetedRefinementRunRecord.id
                == TargetedRefinementCandidateRecord.run_id,
            )
            .where(
                TargetedRefinementRunRecord.predictive_edge_run_id
                == predictive_edge_run_id,
                TargetedRefinementCandidateRecord.id == candidate_id,
            )
        ),
        TargetedRefinementNotFoundError("targeted candidate was not found"),
    )
    return TargetedCandidateDetailResponse(
        **_response(row).model_dump(), run_id=row.run_id
    )
