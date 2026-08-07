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
ECONOMIC_POLICY_SEARCH_V2_PRESET = "economic_policy_search_v2"
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
ECONOMIC_V2_CONTRACT_VERSION = 2
ECONOMIC_V2_POLICY_VERSION = 2
ECONOMIC_V2_CLASSIFICATION_VERSION = 2
ECONOMIC_V2_PROMOTION_POLICY = "economic_successive_halving_v2"
ECONOMIC_V2_STRATEGY_POLICY_VERSION = 2
ECONOMIC_V2_QUALIFICATION_VERSION = 1
ECONOMIC_V2_SEED = 1729
ECONOMIC_V2_DIVERSIFIED_FAMILIES = 432
ECONOMIC_V2_MAX_FAMILIES_PER_SOURCE = 600
ECONOMIC_V2_DEVELOPMENT_START = "2020-01-01T00:00:00+00:00"
ECONOMIC_V2_DEVELOPMENT_END = "2024-01-01T00:00:00+00:00"
ECONOMIC_V2_QUALIFICATION_END = "2025-01-01T00:00:00+00:00"
ECONOMIC_V2_RESERVED_FINAL_END = "2026-01-01T00:00:00+00:00"
ECONOMIC_V2_HORIZONS_SECONDS = tuple(
    hours * 3600 for hours in (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72)
)
ECONOMIC_V2_FEATURE_VARIANTS = ("all", "top_k_16", "top_k_32", "top_k_64")
ECONOMIC_V2_CONFIDENCE_FRACTIONS = tuple(
    Decimal(item) for item in ("0.05", "0.10", "0.20", "0.30", "0.40")
)
ECONOMIC_V2_HOLDING_MULTIPLIERS = tuple(Decimal(item) for item in ("0.5", "1.0", "1.5"))
ECONOMIC_V2_REGIME_POLICIES = (
    "all_regimes",
    "trend_aligned",
    "volatility_middle_60",
)


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
    QUALIFIED_CANDIDATES_FOUND = "qualified_candidates_found"
    NO_QUALIFIED_CANDIDATE = "no_qualified_candidate"


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
    preset: Literal["exhaustive_ohlcv_v1", "economic_policy_search_v2"] = (
        "exhaustive_ohlcv_v1"
    )
    seed_family_ids: tuple[UUID, ...] = ()
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
        if len(set(self.seed_family_ids)) != len(self.seed_family_ids):
            raise ValueError("seed family IDs must be unique")
        if len(self.seed_family_ids) > 10:
            raise ValueError("at most ten seed families are supported")
        if self.preset == CAMPAIGN_PRESET and self.seed_family_ids:
            raise ValueError(
                "seed families are supported only by economic_policy_search_v2"
            )
        minimum = len(self.source_screening_run_ids) * (
            FAMILIES_PER_INSTRUMENT
            if self.preset == CAMPAIGN_PRESET
            else ECONOMIC_V2_DIVERSIFIED_FAMILIES
        )
        if self.maximum_evaluations < minimum:
            raise ValueError(
                f"maximum_evaluations cannot cover stage 1; exact minimum is {minimum}"
            )
        return self

    def normalized_payload(self) -> dict[str, object]:
        if self.preset == ECONOMIC_POLICY_SEARCH_V2_PRESET:
            return {
                "classification_version": ECONOMIC_V2_CLASSIFICATION_VERSION,
                "contract_version": ECONOMIC_V2_CONTRACT_VERSION,
                "development_period": [
                    ECONOMIC_V2_DEVELOPMENT_START,
                    ECONOMIC_V2_DEVELOPMENT_END,
                ],
                "fee_rate": str(CAMPAIGN_FEE_RATE),
                "maximum_evaluations": self.maximum_evaluations,
                "maximum_runtime_seconds": self.maximum_runtime_seconds,
                "policy_version": ECONOMIC_V2_POLICY_VERSION,
                "preset": self.preset,
                "promotion_policy": ECONOMIC_V2_PROMOTION_POLICY,
                "qualification_period": [
                    ECONOMIC_V2_DEVELOPMENT_END,
                    ECONOMIC_V2_QUALIFICATION_END,
                ],
                "qualification_version": ECONOMIC_V2_QUALIFICATION_VERSION,
                "reserved_final_period": [
                    ECONOMIC_V2_QUALIFICATION_END,
                    ECONOMIC_V2_RESERVED_FINAL_END,
                ],
                "seed": ECONOMIC_V2_SEED,
                "seed_family_ids": sorted(str(item) for item in self.seed_family_ids),
                "slippage_rate": str(CAMPAIGN_SLIPPAGE_RATE),
                "source_screening_run_ids": sorted(
                    str(item) for item in self.source_screening_run_ids
                ),
                "stage_trials": list(STAGE_TRIALS),
                "strategy_policy": {
                    "confidence_fractions": [
                        str(item) for item in ECONOMIC_V2_CONFIDENCE_FRACTIONS
                    ],
                    "holding_multipliers": [
                        str(item) for item in ECONOMIC_V2_HOLDING_MULTIPLIERS
                    ],
                    "regime_policies": list(ECONOMIC_V2_REGIME_POLICIES),
                    "version": ECONOMIC_V2_STRATEGY_POLICY_VERSION,
                },
            }
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


class EconomicPolicySearchV2Contract(BaseModel):
    model_config = ConfigDict(frozen=True)
    preset: str = ECONOMIC_POLICY_SEARCH_V2_PRESET
    contract_version: int = ECONOMIC_V2_CONTRACT_VERSION
    policy_version: int = ECONOMIC_V2_POLICY_VERSION
    classification_version: int = ECONOMIC_V2_CLASSIFICATION_VERSION
    qualification_version: int = ECONOMIC_V2_QUALIFICATION_VERSION
    promotion_policy: str = ECONOMIC_V2_PROMOTION_POLICY
    diversified_families_per_source: int = ECONOMIC_V2_DIVERSIFIED_FAMILIES
    maximum_families_per_source: int = ECONOMIC_V2_MAX_FAMILIES_PER_SOURCE
    horizons_seconds: tuple[int, ...] = ECONOMIC_V2_HORIZONS_SECONDS
    feature_variants: tuple[str, ...] = ECONOMIC_V2_FEATURE_VARIANTS
    entry_confidence_fractions: tuple[Decimal, ...] = ECONOMIC_V2_CONFIDENCE_FRACTIONS
    holding_multipliers: tuple[Decimal, ...] = ECONOMIC_V2_HOLDING_MULTIPLIERS
    regime_policies: tuple[str, ...] = ECONOMIC_V2_REGIME_POLICIES
    policy_grid_cardinality: int = 45
    tracks: tuple[str, ...] = ("diversified", "seeded_local")
    minimum_stage_one_evaluations_per_source: int = ECONOMIC_V2_DIVERSIFIED_FAMILIES
    stage_gates: dict[str, dict[str, object]] = Field(
        default_factory=lambda: {
            "stage_1": {
                "minimum_observations": 30,
                "minimum_positive_net_folds": 2,
                "alternative_median_return_strictly_above": "-0.0015",
                "minimum_worst_fold_return": "-0.10",
            },
            "stage_2": {
                "minimum_observations": 50,
                "minimum_positive_net_folds": 3,
                "minimum_folds_beating_buy_and_hold": 2,
                "median_return_strictly_above": "-0.0005",
                "minimum_worst_fold_return": "-0.075",
                "maximum_inactive_folds": 3,
            },
            "stage_3": {
                "minimum_observations": 100,
                "minimum_positive_net_folds": 4,
                "minimum_folds_beating_buy_and_hold": 3,
                "median_return_strictly_above": "0",
                "minimum_worst_fold_return": "-0.05",
                "minimum_round_trips": 10,
                "maximum_inactive_folds": 2,
            },
            "development_candidate": {
                "minimum_observations": 150,
                "minimum_positive_net_folds": 5,
                "minimum_folds_beating_buy_and_hold": 4,
                "minimum_median_return": "0.001",
                "minimum_worst_fold_return": "-0.02",
                "minimum_round_trips": 20,
                "maximum_inactive_folds": 1,
                "maximum_drawdown": "0.25",
                "stressed_returns": "positive_at_1.5x_and_2x",
            },
        }
    )
    qualification_gates: dict[str, object] = Field(
        default_factory=lambda: {
            "minimum_round_trips": 20,
            "minimum_base_return": "0.05",
            "minimum_stressed_returns": "strictly_positive",
            "maximum_drawdown": "0.20",
            "minimum_positive_quarters": 3,
            "minimum_quarter_return": "-0.05",
        }
    )
    development_period: tuple[str, str] = (
        ECONOMIC_V2_DEVELOPMENT_START,
        ECONOMIC_V2_DEVELOPMENT_END,
    )
    qualification_period: tuple[str, str] = (
        ECONOMIC_V2_DEVELOPMENT_END,
        ECONOMIC_V2_QUALIFICATION_END,
    )
    reserved_final_period: tuple[str, str] = (
        ECONOMIC_V2_QUALIFICATION_END,
        ECONOMIC_V2_RESERVED_FINAL_END,
    )


class CampaignContractsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    response_version: int = 2
    legacy: CampaignContract = Field(default_factory=CampaignContract)
    presets: tuple[CampaignContract | EconomicPolicySearchV2Contract, ...] = Field(
        default_factory=lambda: (CampaignContract(), EconomicPolicySearchV2Contract())
    )
