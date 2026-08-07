import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

from app.ml.screening.config import ScreeningEngine

from .contracts import (
    ML_REFINEMENT_CONTRACT_VERSION,
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


def planned_evaluation_from_persisted_values(
    *,
    fingerprint: str,
    engine: str,
    future_duration_seconds: int,
    objective: str,
    hyperparameters: Mapping[str, object],
    feature_policy: Mapping[str, object],
) -> PlannedEvaluation:
    """Materialize and validate a session-independent persisted evaluation."""

    parameters: list[tuple[str, bool | int | float]] = []
    for key, value in hyperparameters.items():
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"invalid frozen hyperparameter: {key}")
        parameters.append((str(key), value))
    variables = feature_policy.get("variables")
    if variables is not None and not isinstance(variables, list):
        raise ValueError("invalid frozen feature variables")
    if isinstance(variables, list) and any(
        not isinstance(item, str) for item in variables
    ):
        raise ValueError("invalid frozen feature variables")
    strategy = feature_policy.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("invalid frozen feature strategy")
    top_k = feature_policy.get("top_k")
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
    ):
        raise ValueError("invalid frozen feature top_k")
    return PlannedEvaluation(
        fingerprint,
        ScreeningEngine(engine),
        future_duration_seconds,
        PredictionObjective(objective),
        tuple(sorted(parameters)),
        strategy,
        tuple(variables) if isinstance(variables, list) else None,
        top_k,
    )


@dataclass(frozen=True, slots=True)
class RefinementPlan:
    evaluations: tuple[PlannedEvaluation, ...]
    total_combinations: int
    omitted_by_trial_budget: int
    uncovered_engines: tuple[str, ...]
    uncovered_objectives: tuple[str, ...]
    uncovered_durations: tuple[int, ...]
    uncovered_feature_policies: tuple[str, ...]


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
    suffix_order: list[tuple[int, int, int]] = []
    for ordinal in range(max(len(objectives), len(durations), len(features))):
        item = (
            ordinal % len(objectives),
            ordinal % len(durations),
            ordinal % len(features),
        )
        if item not in suffix_order:
            suffix_order.append(item)
    for objective_index in range(len(objectives)):
        for duration_index in range(len(durations)):
            for feature_index in range(len(features)):
                item = (objective_index, duration_index, feature_index)
                if item not in suffix_order:
                    suffix_order.append(item)
    for block_index, (block, quota) in enumerate(zip(blocks, quotas, strict=True)):
        randomizer = random.Random(request.deterministic_seed + block_index)
        parameter_offset = randomizer.randrange(block.parameter_count)
        parameter_step = randomizer.randrange(1, block.parameter_count + 1)
        while math.gcd(parameter_step, block.parameter_count) != 1:
            parameter_step = parameter_step % block.parameter_count + 1
        for ordinal in range(quota):
            objective_index, duration_index, feature_index = suffix_order[
                ordinal % suffix_count
            ]
            remainder = ordinal // suffix_count
            parameter_index = (
                parameter_offset + remainder * parameter_step
            ) % block.parameter_count
            parameters = _decode_parameters(
                flat_index=parameter_index,
                names=block.names,
                dimensions=block.dimensions,
            )
            strategy, variables, top_k = features[feature_index]
            duration = durations[duration_index]
            objective = objectives[objective_index]
            payload: dict[str, object] = {
                "contract_version": ML_REFINEMENT_CONTRACT_VERSION,
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
    selected_engines = {item.engine.value for item in candidates}
    selected_objectives = {item.objective.value for item in candidates}
    selected_durations = {item.future_duration_seconds for item in candidates}
    selected_features = {
        f"{item.feature_strategy}:{item.top_k or ''}:{item.explicit_variables or ''}"
        for item in candidates
    }
    requested_features = {
        f"{strategy}:{top_k or ''}:{variables or ''}"
        for strategy, variables, top_k in features
    }
    return RefinementPlan(
        tuple(candidates),
        total,
        max(0, total - len(candidates)),
        tuple(sorted({item.value for item in request.engines} - selected_engines)),
        tuple(
            sorted(
                {item.value for item in request.prediction_objectives}
                - selected_objectives
            )
        ),
        tuple(sorted(set(request.future_durations_seconds) - selected_durations)),
        tuple(sorted(requested_features - selected_features)),
    )
