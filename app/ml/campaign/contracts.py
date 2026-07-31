import hashlib
import json
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ml.contract import FUTURE_DURATIONS
from app.ml.refinement.contracts import (
    CatBoostSearchSpace,
    LightGBMSearchSpace,
    ParameterGrid,
    PredictionObjective,
    RidgeSearchSpace,
)
from app.ml.screening.config import ScreeningEngine

CAMPAIGN_CONTRACT_VERSION = 1
CAMPAIGN_POLICY_VERSION = 1
CAMPAIGN_CLASSIFICATION_VERSION = 1
CAMPAIGN_PRESET = "exhaustive_ohlcv_v1"
PROMOTION_POLICY = "economic_successive_halving_v1"
STAGE_TRIALS = (1, 4, 16, 64)
CAMPAIGN_SEED = 1729
CAMPAIGN_FEE_RATE = Decimal("0.001")
CAMPAIGN_SLIPPAGE_RATE = Decimal("0.0005")
CAMPAIGN_CONFIDENCE_BUCKETS = (Decimal("0.05"), Decimal("0.10"), Decimal("0.20"))
DEFAULT_MAXIMUM_EVALUATIONS = 50_000
DEFAULT_MAXIMUM_RUNTIME_SECONDS = 432_000
MAXIMUM_EVALUATIONS = 100_000
MAXIMUM_RUNTIME_SECONDS = 604_800
FAMILIES_PER_INSTRUMENT = 3600


def campaign_search_spaces() -> tuple[
    RidgeSearchSpace | LightGBMSearchSpace | CatBoostSearchSpace, ...
]:
    def values(*items: bool | int | float) -> ParameterGrid:
        return ParameterGrid(values=items)

    return (
        RidgeSearchSpace(
            engine=ScreeningEngine.RIDGE,
            alpha=values(0.01, 0.1, 1, 10, 100),
            fit_intercept=values(True, False),
        ),
        LightGBMSearchSpace(
            engine=ScreeningEngine.LIGHTGBM,
            learning_rate=values(0.02, 0.05, 0.1),
            n_estimators=values(120, 250, 500),
            max_depth=values(-1, 4, 6, 8),
            num_leaves=values(15, 31, 63),
            min_child_samples=values(20, 50, 100),
            subsample=values(0.7, 1),
            colsample_bytree=values(0.7, 1),
            reg_alpha=values(0, 0.5, 2),
            reg_lambda=values(0, 1, 5),
        ),
        CatBoostSearchSpace(
            engine=ScreeningEngine.CATBOOST,
            learning_rate=values(0.02, 0.05, 0.1),
            iterations=values(120, 250, 500),
            depth=values(4, 5, 6, 7),
            l2_leaf_reg=values(1, 3, 6),
            random_strength=values(0, 0.5, 1),
            bagging_temperature=values(0, 0.5, 1),
        ),
    )


class CampaignStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CampaignOutcome(StrEnum):
    COVERAGE_INCOMPLETE = "coverage_incomplete"
    NO_STRONG_CANDIDATE = "no_strong_candidate"
    SEARCH_BUDGET_EXHAUSTED = "search_budget_exhausted"
    STRONG_CANDIDATES_FOUND = "strong_candidates_found"


class FamilyState(StrEnum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    SURVIVED = "survived"
    DISCARDED = "discarded"
    STRONG_DEVELOPMENT_CANDIDATE = "strong_development_candidate"


class FeatureVariant(StrEnum):
    ALL = "all"
    TOP_16 = "top_k_16"
    TOP_32 = "top_k_32"
    TOP_64 = "top_k_64"
    TOP_96 = "top_k_96"

    @property
    def top_k(self) -> int | None:
        return None if self is FeatureVariant.ALL else int(self.value.rsplit("_", 1)[1])


class CampaignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_screening_run_ids: tuple[UUID, ...]
    preset: Literal["exhaustive_ohlcv_v1"] = "exhaustive_ohlcv_v1"
    maximum_evaluations: int = Field(
        default=DEFAULT_MAXIMUM_EVALUATIONS, ge=1, le=MAXIMUM_EVALUATIONS
    )
    maximum_runtime_seconds: int = Field(
        default=DEFAULT_MAXIMUM_RUNTIME_SECONDS, ge=1, le=MAXIMUM_RUNTIME_SECONDS
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "CampaignRequest":
        if not self.source_screening_run_ids:
            raise ValueError("at least one source screening run is required")
        if len(set(self.source_screening_run_ids)) != len(
            self.source_screening_run_ids
        ):
            raise ValueError("source screening run IDs must be unique")
        minimum = len(self.source_screening_run_ids) * FAMILIES_PER_INSTRUMENT
        if self.maximum_evaluations < minimum:
            raise ValueError(
                f"maximum_evaluations cannot cover stage 1; exact minimum is {minimum}"
            )
        return self

    def normalized_payload(self) -> dict[str, object]:
        return {
            "classification_version": CAMPAIGN_CLASSIFICATION_VERSION,
            "confidence_buckets": [str(item) for item in CAMPAIGN_CONFIDENCE_BUCKETS],
            "contract_version": CAMPAIGN_CONTRACT_VERSION,
            "fee_rate": str(CAMPAIGN_FEE_RATE),
            "maximum_evaluations": self.maximum_evaluations,
            "maximum_runtime_seconds": self.maximum_runtime_seconds,
            "policy_version": CAMPAIGN_POLICY_VERSION,
            "preset": self.preset,
            "promotion_policy": PROMOTION_POLICY,
            "seed": CAMPAIGN_SEED,
            "slippage_rate": str(CAMPAIGN_SLIPPAGE_RATE),
            "source_screening_run_ids": sorted(
                str(item) for item in self.source_screening_run_ids
            ),
            "stage_trials": list(STAGE_TRIALS),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.normalized_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class CampaignContract(BaseModel):
    model_config = ConfigDict(frozen=True)
    preset: str = CAMPAIGN_PRESET
    contract_version: int = CAMPAIGN_CONTRACT_VERSION
    policy_version: int = CAMPAIGN_POLICY_VERSION
    classification_version: int = CAMPAIGN_CLASSIFICATION_VERSION
    promotion_policy: str = PROMOTION_POLICY
    engines: tuple[ScreeningEngine, ...] = tuple(ScreeningEngine)
    future_durations_seconds: tuple[int, ...] = tuple(
        int(item.total_seconds()) for item in FUTURE_DURATIONS
    )
    objectives: tuple[PredictionObjective, ...] = tuple(PredictionObjective)
    feature_variants: tuple[FeatureVariant, ...] = tuple(FeatureVariant)
    stage_trials: tuple[int, ...] = STAGE_TRIALS
    confidence_buckets: tuple[Decimal, ...] = CAMPAIGN_CONFIDENCE_BUCKETS
    fee_rate: Decimal = CAMPAIGN_FEE_RATE
    slippage_rate: Decimal = CAMPAIGN_SLIPPAGE_RATE
    deterministic_seed: int = CAMPAIGN_SEED
    families_per_instrument: int = FAMILIES_PER_INSTRUMENT
    default_maximum_evaluations: int = DEFAULT_MAXIMUM_EVALUATIONS
    default_maximum_runtime_seconds: int = DEFAULT_MAXIMUM_RUNTIME_SECONDS
