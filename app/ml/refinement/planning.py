import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass

from app.ml.screening.config import ScreeningEngine

from .contracts import (
    AllFeatures,
    CatBoostSearchSpace,
    ExplicitFeatures,
    LightGBMSearchSpace,
    MLRefinementRequest,
    PredictionObjective,
    RidgeSearchSpace,
    TopKFeatures,
)


@dataclass(frozen=True, slots=True)
class PlannedEvaluation:
    fingerprint: str
    engine: ScreeningEngine
    future_duration_seconds: int
    objective: PredictionObjective
    hyperparameters: tuple[tuple[str, bool | int | float], ...]
    feature_strategy: str
    explicit_variables: tuple[str, ...] | None
    top_k: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine.value,
            "future_duration_seconds": self.future_duration_seconds,
            "objective": self.objective.value,
            "hyperparameters": dict(self.hyperparameters),
            "feature_strategy": self.feature_strategy,
            "explicit_variables": self.explicit_variables,
            "top_k": self.top_k,
        }


@dataclass(frozen=True, slots=True)
class RefinementPlan:
    evaluations: tuple[PlannedEvaluation, ...]
    total_combinations: int
    omitted_by_trial_budget: int


@dataclass(frozen=True, slots=True)
class _EngineBlock:
    engine: ScreeningEngine
    names: tuple[str, ...]
    dimensions: tuple[tuple[bool | int | float, ...], ...]
    parameter_count: int
    combination_count: int


def _feature_variants(
    request: MLRefinementRequest,
) -> tuple[tuple[str, tuple[str, ...] | None, int | None], ...]:
    variants: list[tuple[str, tuple[str, ...] | None, int | None]] = []
    for policy in request.feature_selection.strategies:
        if isinstance(policy, AllFeatures):
            variants.append(("all", None, None))
        elif isinstance(policy, ExplicitFeatures):
            variants.append(("explicit", tuple(sorted(policy.variables)), None))
        else:
            assert isinstance(policy, TopKFeatures)
            variants.extend(("top_k", None, value) for value in sorted(policy.values))
    return tuple(sorted(variants, key=lambda item: json.dumps(item)))


def _parameter_dimensions(
    space: RidgeSearchSpace | LightGBMSearchSpace | CatBoostSearchSpace,
) -> tuple[tuple[str, ...], tuple[tuple[bool | int | float, ...], ...], int]:
    fields = [name for name in type(space).model_fields if name != "engine"]
    names = tuple(sorted(fields))
    dimensions = tuple(tuple(getattr(space, name).expand()) for name in names)
    size = 1
    for dimension in dimensions:
        size *= len(dimension)
    return names, dimensions, size


def _decode_parameters(
    *,
    flat_index: int,
    names: tuple[str, ...],
    dimensions: tuple[tuple[bool | int | float, ...], ...],
) -> tuple[tuple[str, bool | int | float], ...]:
    values: list[tuple[str, bool | int | float]] = []
    remainder = flat_index
    for name, dimension in reversed(tuple(zip(names, dimensions, strict=True))):
        remainder, index = divmod(remainder, len(dimension))
        values.append((name, dimension[index]))
    return tuple(sorted(values))


def _fingerprint(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_refinement_plan(request: MLRefinementRequest) -> RefinementPlan:
    features = _feature_variants(request)
    spaces = {item.engine: item for item in request.search_space}
    durations = tuple(sorted(request.future_durations_seconds))
    objectives = tuple(sorted(request.prediction_objectives, key=str))
    suffix_count = len(durations) * len(objectives) * len(features)
    blocks: list[_EngineBlock] = []
    for engine in sorted(request.engines, key=str):
        names, dimensions, parameter_count = _parameter_dimensions(spaces[engine])
        blocks.append(
            _EngineBlock(
                engine,
                names,
                dimensions,
                parameter_count,
                parameter_count * suffix_count,
            )
        )
    total = sum(block.combination_count for block in blocks)

    # Allocate the bounded trial budget across engines before sampling. Sampling
    # integer indexes from range objects keeps memory proportional to the budget,
    # not to the potentially enormous Cartesian search space.
    quotas = [0] * len(blocks)
    remaining = min(total, request.maximum_trials)
    while remaining:
        progressed = False
        for index, block in enumerate(blocks):
            if quotas[index] < block.combination_count and remaining:
                quotas[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    candidates: list[PlannedEvaluation] = []
    for block_index, (block, quota) in enumerate(zip(blocks, quotas, strict=True)):
        indexes: Iterable[int]
        if quota == block.combination_count:
            indexes = range(block.combination_count)
        else:
            randomizer = random.Random(request.deterministic_seed + block_index)
            indexes = sorted(randomizer.sample(range(block.combination_count), quota))
        for flat_index in indexes:
            parameter_index, suffix_index = divmod(flat_index, suffix_count)
            duration_index, suffix_index = divmod(
                suffix_index, len(objectives) * len(features)
            )
            objective_index, feature_index = divmod(suffix_index, len(features))
            parameters = _decode_parameters(
                flat_index=parameter_index,
                names=block.names,
                dimensions=block.dimensions,
            )
            strategy, variables, top_k = features[feature_index]
            duration = durations[duration_index]
            objective = objectives[objective_index]
            payload: dict[str, object] = {
                "contract_version": 1,
                "engine": block.engine.value,
                "duration": duration,
                "objective": objective.value,
                "hyperparameters": dict(parameters),
                "feature_strategy": strategy,
                "variables": variables,
                "top_k": top_k,
                "fee_rate": str(request.fee_rate),
                "slippage_rate": str(request.slippage_rate),
                "confidence_buckets": [
                    str(item) for item in request.confidence_buckets
                ],
                "seed": request.deterministic_seed,
            }
            candidates.append(
                PlannedEvaluation(
                    _fingerprint(payload),
                    block.engine,
                    duration,
                    objective,
                    parameters,
                    strategy,
                    variables,
                    top_k,
                )
            )
    candidates.sort(key=lambda item: item.fingerprint)
    return RefinementPlan(tuple(candidates), total, max(0, total - len(candidates)))
