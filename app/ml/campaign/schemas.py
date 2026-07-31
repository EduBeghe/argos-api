from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class CampaignResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    experiment_key: str
    status: str
    outcome: str | None
    contract_version: int
    policy_version: int
    classification_version: int
    maximum_evaluations: int
    maximum_runtime_seconds: int
    families_total: int
    families_covered: int
    families_uncovered: int
    evaluations_computed: int
    evaluations_reused: int
    current_stage: int
    cancellation_requested_at: datetime | None
    first_claimed_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None
    failure_type: str | None
    failure_message: str | None
    created_at: datetime


class CampaignEnqueueResponse(BaseModel):
    created: bool
    campaign: CampaignResponse
    location: str


class CampaignPage(BaseModel):
    items: tuple[CampaignResponse, ...]
    total: int
    offset: int
    limit: int
    has_more: bool


class FamilyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    campaign_id: UUID
    source_screening_run_id: UUID
    fingerprint: str
    asset: str
    provider: str
    market_type: str
    interval: str
    engine: str
    future_duration_seconds: int
    objective: str
    feature_variant: str
    state: str
    current_stage: int
    evaluations_planned: int
    evaluations_completed: int
    evaluations_reused: int
    best_evaluation_id: UUID | None
    best_metrics: dict[str, Any] | None
    decision_reason: str | None
    deterministic_ordinal: int


class FamilyPage(BaseModel):
    items: tuple[FamilyResponse, ...]
    total: int
    offset: int
    limit: int
    has_more: bool

