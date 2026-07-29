from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.screening.config import ScreeningEngine
from app.ml.screening.evidence import EvidenceClassification


class ScreeningRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    experiment_key: str
    status: str
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    dataset_fingerprint: str
    screening_version: int
    experiment_contract_version: int
    variable_registry_version: int
    missing_data_policy_version: int
    requested_engines: list[str]
    requested_future_durations: list[int]
    library_versions: dict[str, str]
    deterministic_seed: int
    configuration: dict[str, Any]
    evaluations_computed: int
    evaluations_reused: int
    results_generated: int
    pareto_retained_count: int
    started_at: datetime
    completed_at: datetime | None
    elapsed_seconds: Decimal | None
    failure_type: str | None
    failure_message: str | None


class ScreeningRunPage(BaseModel):
    items: list[ScreeningRunSummary]
    total: int
    limit: int
    offset: int
    has_more: bool


class ClassificationCounts(BaseModel):
    promising: int = 0
    insufficient: int = 0
    discarded: int = 0
    incomplete: int = 0


class ScreeningRunDetail(BaseModel):
    run: ScreeningRunSummary
    classification_counts: ClassificationCounts
    engine_counts: dict[ScreeningEngine, int]
    pareto_retained_count: int


class ScreeningResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    run_id: UUID
    engine: ScreeningEngine
    future_duration_seconds: int
    folds_evaluated: int
    total_validation_samples: int
    median_mae_improvement_over_zero: Decimal
    worst_fold_mae_improvement_over_zero: Decimal
    median_directional_accuracy: Decimal
    worst_fold_directional_accuracy: Decimal
    median_pearson_correlation: Decimal | None
    median_spearman_correlation: Decimal | None
    folds_beating_zero: int
    folds_with_positive_spearman: int
    total_training_time_seconds: Decimal
    pareto_retained: bool
    display_rank: int
    evidence_classification: EvidenceClassification
    evidence_classification_version: int
    classification_explanation: str
    failed_classification_rules: list[str]
    created_at: datetime


class ScreeningResultPage(BaseModel):
    items: list[ScreeningResultResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class FoldEvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    run_id: UUID
    engine: ScreeningEngine
    future_duration_seconds: int
    fold_name: str
    training_row_count: int
    validation_row_count: int
    sample_count: int
    mean_absolute_error: Decimal
    root_mean_squared_error: Decimal
    mean_prediction: Decimal
    mean_actual_return: Decimal
    pearson_correlation: Decimal | None
    spearman_correlation: Decimal | None
    directional_accuracy: Decimal
    zero_prediction_mae: Decimal
    mae_improvement_over_zero: Decimal
    positive_prediction_count: int
    negative_prediction_count: int
    elapsed_seconds: Decimal
    created_at: datetime


class ScreeningResultDetail(BaseModel):
    result: ScreeningResultResponse
    fold_evaluations: list[FoldEvaluationResponse]

