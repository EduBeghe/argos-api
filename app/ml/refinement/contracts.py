import hashlib
import json
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ml.contract import FUTURE_DURATIONS
from app.ml.registry import VARIABLE_REGISTRY
from app.ml.screening.config import ScreeningEngine

ML_REFINEMENT_CONTRACT_VERSION = 2
ML_REFINEMENT_CLASSIFICATION_VERSION = 2
MAXIMUM_TRIAL_BUDGET = 10_000
MAXIMUM_RUNTIME_SECONDS = 86_400


class PredictionObjective(StrEnum):
    FUTURE_RETURN_REGRESSION = "future_return_regression"
    FUTURE_DIRECTION_CLASSIFICATION = "future_direction_classification"
    COST_ADJUSTED_ACTION_CLASSIFICATION = "cost_adjusted_action_classification"


class RefinementStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RefinementClassification(StrEnum):
    PROMISING = "promising_for_strategy_simulation"
    INSUFFICIENT = "insufficient_evidence"
    DISCARDED = "discarded"
    INCOMPLETE = "incomplete"


Scalar = bool | int | float


class ParameterGrid(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    values: tuple[Scalar, ...] | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None

    @model_validator(mode="after")
    def validate_form(self) -> "ParameterGrid":
        explicit = self.values is not None
        ranged = any(
            value is not None for value in (self.minimum, self.maximum, self.step)
        )
        if explicit == ranged:
            raise ValueError("parameter requires exactly one of values or range")
        if explicit:
            if not self.values:
                raise ValueError("parameter values must not be empty")
            if len(set(self.values)) != len(self.values):
                raise ValueError("parameter values must be unique")
            return self
        if self.minimum is None or self.maximum is None or self.step is None:
            raise ValueError("range requires minimum, maximum and step")
        if self.minimum > self.maximum:
            raise ValueError("parameter range is inverted")
        if self.step <= 0:
            raise ValueError("parameter step must be positive")
        return self

    def expand(self) -> tuple[Scalar, ...]:
        if self.values is not None:
            return tuple(sorted(self.values, key=lambda item: (str(type(item)), item)))
        assert self.minimum is not None
        assert self.maximum is not None
        assert self.step is not None
        values: list[Scalar] = []
        current = self.minimum
        while current <= self.maximum + abs(float(self.step)) * 1e-12:
            values.append(current)
            current = current + self.step
        return tuple(values)


def _numbers(grid: ParameterGrid, name: str) -> tuple[float, ...]:
    values = grid.expand()
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{name} requires numeric values")
    return tuple(float(value) for value in values)


def _positive(grid: ParameterGrid, name: str, *, allow_zero: bool = False) -> None:
    values = _numbers(grid, name)
    if any(value < 0 if allow_zero else value <= 0 for value in values):
        raise ValueError(f"{name} values are outside the supported range")


def _fraction(grid: ParameterGrid, name: str) -> None:
    if any(not 0 < value <= 1 for value in _numbers(grid, name)):
        raise ValueError(f"{name} must be in (0, 1]")


def _integers(grid: ParameterGrid, name: str, *, minimum: int = 1) -> None:
    values = grid.expand()
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
        for value in values
    ):
        raise ValueError(f"{name} requires integers greater than or equal to {minimum}")


class RidgeSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine: Literal[ScreeningEngine.RIDGE]
    alpha: ParameterGrid
    fit_intercept: ParameterGrid

    @model_validator(mode="after")
    def validate_parameters(self) -> "RidgeSearchSpace":
        _positive(self.alpha, "alpha")
        if any(not isinstance(value, bool) for value in self.fit_intercept.expand()):
            raise ValueError("fit_intercept requires boolean values")
        return self


class LightGBMSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine: Literal[ScreeningEngine.LIGHTGBM]
    learning_rate: ParameterGrid
    n_estimators: ParameterGrid
    max_depth: ParameterGrid
    num_leaves: ParameterGrid
    min_child_samples: ParameterGrid
    subsample: ParameterGrid
    colsample_bytree: ParameterGrid
    reg_alpha: ParameterGrid
    reg_lambda: ParameterGrid

    @model_validator(mode="after")
    def validate_parameters(self) -> "LightGBMSearchSpace":
        for name in (
            "learning_rate",
            "n_estimators",
            "num_leaves",
            "min_child_samples",
        ):
            _positive(getattr(self, name), name)
        for name in ("n_estimators", "num_leaves", "min_child_samples"):
            _integers(getattr(self, name), name)
        _integers(self.max_depth, "max_depth", minimum=-1)
        _fraction(self.subsample, "subsample")
        _fraction(self.colsample_bytree, "colsample_bytree")
        _positive(self.reg_alpha, "reg_alpha", allow_zero=True)
        _positive(self.reg_lambda, "reg_lambda", allow_zero=True)
        return self


class CatBoostSearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine: Literal[ScreeningEngine.CATBOOST]
    learning_rate: ParameterGrid
    iterations: ParameterGrid
    depth: ParameterGrid
    l2_leaf_reg: ParameterGrid
    random_strength: ParameterGrid
    bagging_temperature: ParameterGrid

    @model_validator(mode="after")
    def validate_parameters(self) -> "CatBoostSearchSpace":
        for name in ("learning_rate", "iterations", "depth"):
            _positive(getattr(self, name), name)
        _integers(self.iterations, "iterations")
        _integers(self.depth, "depth")
        for name in ("l2_leaf_reg", "random_strength", "bagging_temperature"):
            _positive(getattr(self, name), name, allow_zero=True)
        return self


EngineSearchSpace = Annotated[
    RidgeSearchSpace | LightGBMSearchSpace | CatBoostSearchSpace,
    Field(discriminator="engine"),
]


class AllFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy: Literal["all"]


class ExplicitFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy: Literal["explicit"]
    variables: tuple[str, ...]

    @model_validator(mode="after")
    def validate_variables(self) -> "ExplicitFeatures":
        known = {item.name for item in VARIABLE_REGISTRY}
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("explicit variables must be non-empty and unique")
        unknown = set(self.variables) - known
        if unknown:
            raise ValueError(f"unknown variables: {sorted(unknown)}")
        return self


class TopKFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy: Literal["top_k"]
    values: tuple[int, ...]

    @model_validator(mode="after")
    def validate_k(self) -> "TopKFeatures":
        maximum = len(VARIABLE_REGISTRY)
        if (
            not self.values
            or len(set(self.values)) != len(self.values)
            or any(value < 1 or value > maximum for value in self.values)
        ):
            raise ValueError("top_k values must be unique and within the registry")
        return self


FeaturePolicy = Annotated[
    AllFeatures | ExplicitFeatures | TopKFeatures, Field(discriminator="strategy")
]


class FeatureSelectionSearch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategies: tuple[FeaturePolicy, ...]

    @model_validator(mode="after")
    def validate_strategies(self) -> "FeatureSelectionSearch":
        if not self.strategies:
            raise ValueError("feature selection strategies must not be empty")
        serialized = [
            json.dumps(item.model_dump(), sort_keys=True) for item in self.strategies
        ]
        if len(set(serialized)) != len(serialized):
            raise ValueError("feature selection strategies must be unique")
        return self


class EconomicPolicySearchConfiguration(BaseModel):
    """Versioned strategy-policy grid evaluated without refitting a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    confidence_fractions: tuple[Decimal, ...]
    holding_multipliers: tuple[Decimal, ...]
    regime_policies: tuple[str, ...]
    version: int


class MLRefinementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_screening_run_id: UUID
    source_screening_result_id: UUID
    engines: tuple[ScreeningEngine, ...]
    future_durations_seconds: tuple[int, ...]
    search_space: tuple[EngineSearchSpace, ...]
    feature_selection: FeatureSelectionSearch
    prediction_objectives: tuple[PredictionObjective, ...]
    confidence_buckets: tuple[Decimal, ...]
    fee_rate: Decimal = Field(ge=0, le=1)
    slippage_rate: Decimal = Field(ge=0, le=1)
    maximum_trials: int = Field(ge=1, le=MAXIMUM_TRIAL_BUDGET)
    maximum_runtime_seconds: int = Field(ge=1, le=MAXIMUM_RUNTIME_SECONDS)
    deterministic_seed: int
    economic_policy_search: EconomicPolicySearchConfiguration | None = None
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_contract(self) -> "MLRefinementRequest":
        for name, values in (
            ("engines", self.engines),
            ("future durations", self.future_durations_seconds),
            ("prediction objectives", self.prediction_objectives),
            ("confidence buckets", self.confidence_buckets),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be non-empty and unique")
        allowed = {int(item.total_seconds()) for item in FUTURE_DURATIONS}
        if not set(self.future_durations_seconds) <= allowed:
            raise ValueError("future duration is outside the current ML contract")
        if any(value <= 0 or value > 1 for value in self.confidence_buckets):
            raise ValueError("confidence buckets must be fractions in (0, 1]")
        spaces = [item.engine for item in self.search_space]
        if len(set(spaces)) != len(spaces) or set(spaces) != set(self.engines):
            raise ValueError(
                "search_space must define each requested engine exactly once"
            )
        if self.maximum_trials < len(self.engines):
            raise ValueError("maximum_trials must cover every requested engine")
        return self

    def normalized_payload(self) -> dict[str, object]:
        payload = self.model_dump(mode="json", exclude={"notes"})
        payload["engines"] = sorted(payload["engines"])
        payload["future_durations_seconds"] = sorted(
            payload["future_durations_seconds"]
        )
        payload["prediction_objectives"] = sorted(payload["prediction_objectives"])
        payload["confidence_buckets"] = sorted(payload["confidence_buckets"])
        payload["search_space"] = sorted(
            payload["search_space"], key=lambda item: item["engine"]
        )
        payload["feature_selection"]["strategies"] = sorted(
            payload["feature_selection"]["strategies"],
            key=lambda item: json.dumps(item, sort_keys=True),
        )
        if self.economic_policy_search is None:
            payload.pop("economic_policy_search", None)
        return payload

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.normalized_payload(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
