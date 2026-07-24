from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class CompositeSearchSummaryResponse(BaseModel):
    id: UUID
    predictive_edge_run_id: UUID
    conclusion: str
    maximum_depth_reached: int
    feature_pool: tuple[str, ...]
    excluded_features: dict[str, str]
    funnel_by_depth: tuple[dict[str, object], ...]
    total_evaluations: int
    computed: int
    reused: int
    best_depth: int | None
    best_candidate_fingerprint: str | None
    confirmed_candidate_count: int
    dominant_rejection_reason: str | None
    started_at: datetime
    completed_at: datetime


class CompositeCandidateResponse(BaseModel):
    id: UUID
    candidate_fingerprint: str
    parent_fingerprint: str | None
    depth: int
    features: tuple[str, ...]
    signed_weights: tuple[dict[str, object], ...]
    normalization_parameters: tuple[dict[str, object], ...]
    configuration: dict[str, object]
    status: str
    ranking_breakdown: dict[str, object]
    training_metrics: dict[str, object]
    validation_metrics: dict[str, object] | None
    test_metrics: dict[str, object] | None
    passed_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    rejection_reason: str | None
    normalized_distance: str


class CompositeCandidatePageResponse(BaseModel):
    items: tuple[CompositeCandidateResponse, ...]
    total: int
    limit: int
    offset: int


class CompositeCandidateDetailResponse(CompositeCandidateResponse):
    run_id: UUID
