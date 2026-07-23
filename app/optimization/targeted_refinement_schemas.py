from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.optimization.targeted_refinement import (
    TargetedCandidateStatus,
    TargetedRefinementConclusion,
)


class TargetedRefinementSummaryResponse(BaseModel):
    id: UUID
    predictive_edge_run_id: UUID
    conclusion: TargetedRefinementConclusion
    summary: dict[str, object]
    confirmed_candidate_count: int
    best_targeted_candidate_id: UUID | None
    best_targeted_candidate: str | None


class TargetedCandidateResponse(BaseModel):
    id: UUID
    candidate_fingerprint: str
    seed_fingerprint: str
    status: TargetedCandidateStatus
    normalized_distance: Decimal
    configuration: dict[str, object]
    training_metrics: dict[str, object] | None
    validation_metrics: dict[str, object] | None
    test_metrics: dict[str, object] | None
    passed_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    requirement_margins: tuple[dict[str, object], ...]
    rejection_reason: str | None


class TargetedCandidatePageResponse(BaseModel):
    items: tuple[TargetedCandidateResponse, ...]
    total: int
    next_cursor: str | None
    has_more: bool


class TargetedCandidateDetailResponse(TargetedCandidateResponse):
    run_id: UUID
