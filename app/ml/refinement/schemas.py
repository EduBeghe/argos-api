
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.ml.screening.config import ScreeningEngine

from .contracts import PredictionObjective, RefinementClassification, RefinementStatus


class RefinementRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    experiment_key: str
    source_screening_run_id: UUID
    source_screening_result_id: UUID
    status: RefinementStatus
    contract_version: int
    classification_version: int
    dataset_fingerprint: str
    variable_registry_version: int
    missing_data_policy_version: int
    library_versions: dict[str, str]
    normalized_configuration: dict[str, Any]
    notes: str | None
    maximum_trials: int
    maximum_runtime_seconds: int
    combinations_planned: int
    evaluations_computed: int
    evaluations_reused: int
    evaluations_omitted: int
    budget_exhausted: bool
    termination_reason: str | None
    started_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None
    failure_type: str | None
    failure_message: str | None
    elapsed_seconds: Decimal | None
    created_at: datetime


class RefinementRunPage(BaseModel):
    items: list[RefinementRunResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class RefinementRunDetail(BaseModel):
    run: RefinementRunResponse
    classification_counts: dict[RefinementClassification, int]
    engine_counts: dict[ScreeningEngine, int]


class RefinementEvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    run_id: UUID
    reused_from_evaluation_id: UUID | None
    fingerprint: str
    engine: ScreeningEngine
    future_duration_seconds: int
    objective: PredictionObjective
    hyperparameters: dict[str, Any]
    feature_policy: dict[str, Any]
    confidence_buckets: list[str]
    aggregate_metrics: dict[str, Any]
    classification: RefinementClassification
    classification_version: int
    failed_rules: list[str]
    elapsed_seconds: Decimal
    created_at: datetime


class RefinementEvaluationDetail(RefinementEvaluationResponse):
    fold_evidence: list[dict[str, Any]]


class RefinementEvaluationPage(BaseModel):
    items: list[RefinementEvaluationResponse]
    total: int
    limit: int
    offset: int
    has_more: bool
