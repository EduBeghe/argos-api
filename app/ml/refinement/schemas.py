from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, computed_field

from app.ml.screening.config import ScreeningEngine

from .contracts import (
    PredictionObjective,
    RefinementClassification,
    RefinementStatus,
)
from .queue import QueueDisposition


class RefinementVariableResponse(BaseModel):
    name: str
    description: str
    category: str
    lookback_seconds: int
    calculation: str
    calculation_version: int
    higher_is_directional: bool | None


class RefinementLimitsResponse(BaseModel):
    maximum_trials: int
    maximum_runtime_seconds: int
    maximum_notes_length: int


class RefinementDefaultsResponse(BaseModel):
    fee_rate: Decimal
    slippage_rate: Decimal
    deterministic_seed: int


class RefinementContractResponse(BaseModel):
    contract_version: int
    classification_version: int
    variable_registry_version: int
    missing_data_policy_version: int
    supported_engines: list[str]
    supported_prediction_objectives: list[PredictionObjective]
    allowed_future_durations_seconds: list[int]
    variables: list[RefinementVariableResponse]
    validation_limits: RefinementLimitsResponse
    ui_defaults: RefinementDefaultsResponse


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
    evaluations_selected: int
    budget_exhausted: bool
    termination_reason: str | None
    started_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None
    failure_type: str | None
    failure_message: str | None
    worker_id: str | None
    claimed_at: datetime | None
    heartbeat_at: datetime | None
    cancellation_requested_at: datetime | None
    elapsed_seconds: Decimal | None
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def work_completed(self) -> int:
        return min(
            self.evaluations_computed + self.evaluations_reused,
            self.evaluations_selected,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress_fraction(self) -> Decimal | None:
        if self.evaluations_selected == 0:
            return None
        return Decimal(self.work_completed) / Decimal(self.evaluations_selected)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cancellation_requested(self) -> bool:
        return self.cancellation_requested_at is not None


class RefinementEnqueueResponse(BaseModel):
    disposition: QueueDisposition
    run: RefinementRunResponse
    location: str


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
